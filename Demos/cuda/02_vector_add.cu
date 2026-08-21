/* Vector add on the GPU: the full host/device dance (malloc, memcpy, launch),
   plus three things the professor asked every demo to show:
     (a) CORRECTNESS CHECK  — GPU result compared to a CPU reference, PASS/FAIL.
     (b) PARAMETER VARYING  — the coalesced kernel is timed at blockDim
                              128 / 256 / 1024 to show the occupancy effect.
     (c) INEFFICIENT vs OPTIMIZED — a STRIDED (non-coalesced) kernel vs a
                              CONTIGUOUS (coalesced) one, timed with cudaEvents.

   Build: nvcc -O3 -arch=sm_86 -o 02_vector_add 02_vector_add.cu             */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err__), __FILE__, __LINE__);           \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

/* OPTIMIZED: thread i touches element i. Consecutive threads in a warp touch
   consecutive addresses, so the 32 accesses coalesce into one memory
   transaction — full DRAM bandwidth. */
__global__ void vadd_coalesced(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

/* INEFFICIENT: same work, but element index = thread * STRIDE. Consecutive
   threads now touch addresses STRIDE floats apart, so the warp's 32 accesses
   scatter across many cache lines and CANNOT coalesce — many more memory
   transactions for the identical arithmetic.
   FIX: index by the natural thread id (see vadd_coalesced) so a warp reads a
   contiguous run of memory. */
#define STRIDE 32
__global__ void vadd_strided(const float *a, const float *b, float *c, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    /* Remap t -> a scattered index that still covers [0,n) as a permutation. */
    int chunk = n / STRIDE;
    int i = (t % chunk) * STRIDE + (t / chunk);
    if (i < n) c[i] = a[i] + b[i];
}

/* Time a kernel launch (already-configured) over the two access patterns. */
static float time_kernel(void (*which)(const float*,const float*,float*,int),
                         const float *da, const float *db, float *dc, int n,
                         int threads) {
    int blocks = (n + threads - 1) / threads;
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    which<<<blocks, threads>>>(da, db, dc, n);   /* warm-up */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int r = 0; r < 50; r++) which<<<blocks, threads>>>(da, db, dc, n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / 50.0f;                            /* avg ms per launch */
}

int main(void) {
    int n = 1 << 24;                              /* 16.7 M elements */
    size_t bytes = (size_t)n * sizeof(float);
    float *a = (float*)malloc(bytes), *b = (float*)malloc(bytes), *c = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { a[i] = 1.0f + (i & 7); b[i] = 2.0f + (i & 3); }

    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));

    /* ---- (a) CORRECTNESS: run the coalesced kernel once, compare to CPU. ---- */
    int threads = 256, blocks = (n + threads - 1) / threads;
    vadd_coalesced<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double ref = (double)a[i] + (double)b[i];       /* CPU reference */
        double e = fabs((double)c[i] - ref);
        if (e > max_err) max_err = e;
    }
    printf("Correctness (n=%d): %s  (max error %.3e)\n",
           n, max_err < 1e-5 ? "PASS" : "FAIL", max_err);

    /* ---- (b) PARAMETER VARYING: coalesced kernel at 3 block sizes. ---- */
    printf("\nblockDim sweep (coalesced kernel):\n");
    for (int t : {128, 256, 1024}) {
        float ms = time_kernel(vadd_coalesced, da, db, dc, n, t);
        double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;  /* read a,b + write c */
        printf("  blockDim %4d : %.3f ms   %.1f GB/s\n", t, ms, gbps);
    }

    /* ---- (c) INEFFICIENT vs OPTIMIZED at a fixed 256-thread config. ---- */
    float ms_bad  = time_kernel(vadd_strided,   da, db, dc, n, 256);
    float ms_good = time_kernel(vadd_coalesced, da, db, dc, n, 256);
    printf("\nAccess pattern (blockDim 256):\n");
    printf("  strided (non-coalesced): %.3f ms\n", ms_bad);
    printf("  coalesced              : %.3f ms\n", ms_good);
    printf("  speedup (bad/good)     : %.2fx\n", ms_bad / ms_good);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(a); free(b); free(c);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), Xeon w7-2495X 48c:
Correctness (n=16777216): PASS  (max error 0.000e+00)

blockDim sweep (coalesced kernel):
  blockDim  128 : 0.352 ms   571.3 GB/s
  blockDim  256 : 0.353 ms   570.9 GB/s
  blockDim 1024 : 0.355 ms   567.8 GB/s

Access pattern (blockDim 256):
  strided (non-coalesced): 3.740 ms
  coalesced              : 0.353 ms
  speedup (bad/good)     : 10.61x
*/

/* ============================================================================
   STRIDED (non-coalesced) global access -- the UNOPTIMIZED version.
   Split out of the repo's 02_vector_add.cu; the fix is strided/strided_opt.cu.
   ----------------------------------------------------------------------------
   Element index = a scattered remap of the thread id, so consecutive threads
   in a warp touch addresses STRIDE floats apart.  The warp's 32 accesses
   scatter across 32 different 32-byte sectors and CANNOT coalesce: one useful
   word per sector, 7 of 8 words thrown away, for the identical arithmetic.

   In a cuThermo heat map this shows up as sector after sector with a single
   non-zero word.  Note that each word here is touched by exactly ONE warp, so
   the counts read

       Sector k | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |   total = 1

   i.e. a warp-PRIVATE stride.  The paper's Fig. 5(d) additionally has every
   warp in the block hitting the same word column (count 8), which is what a
   whole block walking one column of a row-major matrix produces.

     (a) CORRECTNESS CHECK  — GPU result vs a CPU reference, PASS/FAIL.
     (b) PARAMETER VARYING  — blockDim 128 / 256 / 1024.
     (c) THE INEFFICIENCY   — see above; FIX in strided/strided_opt.cu.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o strided_naive strided_naive.cu                  */
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

#define STRIDE 32
__global__ void vadd_strided(const float *a, const float *b, float *c, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    /* Remap t -> a scattered index that still covers [0,n) as a permutation. */
    int chunk = n / STRIDE;
    int i = (t % chunk) * STRIDE + (t / chunk);
    if (i < n) c[i] = a[i] + b[i];
}

static float time_kernel(const float *da, const float *db, float *dc, int n,
                         int threads) {
    int blocks = (n + threads - 1) / threads;
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    vadd_strided<<<blocks, threads>>>(da, db, dc, n);   /* warm-up */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int r = 0; r < 50; r++) vadd_strided<<<blocks, threads>>>(da, db, dc, n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / 50.0f;
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

    /* ---- (a) CORRECTNESS ---- */
    int threads = 256, blocks = (n + threads - 1) / threads;
    CUDA_CHECK(cudaMemset(dc, 0, bytes));
    vadd_strided<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double ref = (double)a[i] + (double)b[i];
        double e = fabs((double)c[i] - ref);
        if (e > max_err) max_err = e;
    }
    printf("Vector add, STRIDED (non-coalesced) access  [unoptimized]\n");
    printf("Correctness (n=%d): %s  (max error %.3e)\n",
           n, max_err < 1e-5 ? "PASS" : "FAIL", max_err);

    /* ---- (b) PARAMETER VARYING ---- */
    printf("\nblockDim sweep:\n");
    for (int t : {128, 256, 1024}) {
        float ms = time_kernel(da, db, dc, n, t);
        double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;   /* read a,b + write c */
        printf("  blockDim %4d : %.3f ms   %.1f GB/s\n", t, ms, gbps);
    }
    printf("\nCompare with strided/strided_opt.cu (same work, coalesced indexing).\n");

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(a); free(b); free(c);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
Vector add, STRIDED (non-coalesced) access  [unoptimized]
Correctness (n=16777216): PASS  (max error 0.000e+00)

blockDim sweep:
  blockDim  128 : 3.791 ms   53.1 GB/s
  blockDim  256 : 3.721 ms   54.1 GB/s
  blockDim 1024 : 3.611 ms   55.8 GB/s

vs strided/strided_opt.cu at blockDim 256: 3.721 / 0.353 = 10.5x slower.
*/

/* ============================================================================
   STRIDED (non-coalesced) global access -- the OPTIMIZED version.
   Split out of the repo's 02_vector_add.cu; the unoptimized one is
   strided/vadd_naive.cu.
   ----------------------------------------------------------------------------
   Thread i touches element i.  Consecutive threads in a warp touch consecutive
   addresses, so the warp's 32 accesses coalesce: 128 bytes = exactly four
   32-byte sectors, every word of every sector used.  Full DRAM bandwidth.

   In a cuThermo heat map the same memory now reads

       Sector k | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |   total = 1

   -- one warp, every word used, one transaction.  That is what a healthy
   streaming access looks like, and it is the reference to compare all the
   inefficiency patterns against.

   ############################################################################
   #  WHAT CHANGED vs strided/vadd_naive.cu                                #
   ############################################################################

   ONE LINE -- the index expression.  Same arrays, same arithmetic, same launch
   configuration; only how a thread id maps to an element changes:

       vadd_naive.cu:
           int t = blockIdx.x * blockDim.x + threadIdx.x;
           int chunk = n / STRIDE;                        // STRIDE = 32
           int i = (t % chunk) * STRIDE + (t / chunk);    // scattered permutation
           if (i < n) c[i] = a[i] + b[i];

       vadd_opt.cu:
           int i = blockIdx.x * blockDim.x + threadIdx.x; // identity mapping
           if (i < n) c[i] = a[i] + b[i];

   The naive index is still a permutation of [0,n), so both kernels write every
   element exactly once and both PASS the correctness check -- the ONLY
   difference is the order in which memory is touched.  Consecutive lanes in
   vadd_naive.cu land STRIDE floats = 128 bytes apart, one useful word per
   32-byte sector; in vadd_opt.cu they land on consecutive words, so a
   warp's 32 accesses become four full sectors.

   Also removed: the `#define STRIDE 32` and the `chunk` computation, which
   have no purpose once the mapping is the identity.

     (a) CORRECTNESS CHECK  — GPU result vs a CPU reference, PASS/FAIL.
     (b) PARAMETER VARYING  — blockDim 128 / 256 / 1024.
     (c) THE FIX            — index by the natural thread id.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o vadd_opt vadd_opt.cu                        */
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

__global__ void vadd_coalesced(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

static float time_kernel(const float *da, const float *db, float *dc, int n,
                         int threads) {
    int blocks = (n + threads - 1) / threads;
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    vadd_coalesced<<<blocks, threads>>>(da, db, dc, n);   /* warm-up */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int r = 0; r < 50; r++) vadd_coalesced<<<blocks, threads>>>(da, db, dc, n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / 50.0f;
}

int main(void) {
    int n = 1 << 24;                              /* 16.7 M elements */
    /* The scattered index in vadd_naive.cu is a permutation of [0,n) only
       when STRIDE divides n; otherwise the tail elements are never written and
       both files would disagree with the CPU reference. Checked here so the
       constraint fails loudly instead of silently. */
    if (n % 32 != 0) {
        fprintf(stderr, "n (%d) must be a multiple of STRIDE (32)\n", n);
        return 1;
    }
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
    vadd_coalesced<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double ref = (double)a[i] + (double)b[i];
        double e = fabs((double)c[i] - ref);
        if (e > max_err) max_err = e;
    }
    printf("Vector add, COALESCED access  [optimized]\n");
    printf("Correctness (n=%d): %s  (max error %.3e)\n",
           n, max_err < 1e-5 ? "PASS" : "FAIL", max_err);

    /* ---- (b) PARAMETER VARYING ---- */
    printf("\nblockDim sweep:\n");
    for (int t : {128, 256, 1024}) {
        float ms = time_kernel(da, db, dc, n, t);
        double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;   /* read a,b + write c */
        printf("  blockDim %4d : %.3f ms   %.1f GB/s\n", t, ms, gbps);
    }
    printf("\nCompare with strided/vadd_naive.cu (same work, scattered indexing).\n");

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(a); free(b); free(c);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
Vector add, COALESCED access  [optimized]
Correctness (n=16777216): PASS  (max error 0.000e+00)

blockDim sweep:
  blockDim  128 : 0.352 ms   571.7 GB/s
  blockDim  256 : 0.353 ms   570.7 GB/s
  blockDim 1024 : 0.355 ms   567.9 GB/s

vs strided/vadd_naive.cu at blockDim 256: 0.353 vs 3.721 ms = 10.5x faster.
*/

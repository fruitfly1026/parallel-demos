/* ============================================================================
   cuThermo pattern: ABUSE OF SHARED MEMORY -- the OPTIMIZED version.
   [paper Fig. 5(a); Sec. 6.2, PASTA's spt_TTMRankRBNnzKernelSM / Y_shr]
   ----------------------------------------------------------------------------
   PROVENANCE: see smem_abuse/ttm_naive.cu.  Self-contained kernel with
   PASTA's shape, not PASTA's code.

   ############################################################################
   #  WHAT CHANGED vs smem_abuse/ttm_naive.cu                          #
   ############################################################################

   THE ACCUMULATOR MOVED FROM SHARED MEMORY TO REGISTERS.  The arithmetic, the
   loop bounds, the global-memory indexing and the launch configuration are
   all untouched:

       naive:                                     opt:
           __shared__ float Y_shr[BLK][RANK];         float acc[RANK];
           for (r) Y_shr[tid][r] = 0.0f;              #pragma unroll
           __syncthreads();                           for (r) acc[r] = 0.0f;
           for (k) {                                  for (k) {
             float v = X[k*n + i];                      float v = X[k*n + i];
             for (r)                                    #pragma unroll
               Y_shr[tid][r] += v * U[k*RANK+r];        for (r)
           }                                              acc[r] += v * U[k*RANK+r];
           __syncthreads();                           }
           for (r) Y[i*RANK+r] = Y_shr[tid][r];       #pragma unroll
                                                      for (r) Y[i*RANK+r] = acc[r];

   Three things go away:
     1. 16 KB of shared memory per block, returned to L1;
     2. both __syncthreads() barriers, which were synchronising nothing --
        no thread ever read another thread's slot;
     3. NNZ*RANK = 512 shared-memory read-modify-writes per thread, replaced
        by register accumulation.

   The `#pragma unroll` on the RANK loops is required, not cosmetic: acc[] is
   a local array, and nvcc only keeps a local array in registers when every
   index is resolved at compile time.  Without the pragma it spills to local
   memory, which is slower than the shared memory we were trying to escape.
   This is exactly the fix the paper applies to PASTA in Sec. 6.2, where
   Y_shr becomes a scalar local_sum.

   WHAT THE HEAT MAP SHOWS
       The SMEM-tagged line is simply GONE.  There is no cooler version of an
       unshared shared-memory region -- the fix is to stop allocating it.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o ttm_opt ttm_opt.cu
   ========================================================================= */
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

/* Time any launch expression: TIME_KERNEL(ms, reps, k<<<g,b>>>(args...)). */
#define TIME_KERNEL(ms, reps, ...)                                            \
    do {                                                                      \
        cudaEvent_t s__, e__;                                                 \
        cudaEventCreate(&s__); cudaEventCreate(&e__);                         \
        __VA_ARGS__;                                                          \
        CUDA_CHECK(cudaGetLastError());                                       \
        CUDA_CHECK(cudaDeviceSynchronize());                                  \
        cudaEventRecord(s__);                                                 \
        for (int r__ = 0; r__ < (reps); ++r__) { __VA_ARGS__; }               \
        cudaEventRecord(e__); cudaEventSynchronize(e__);                      \
        cudaEventElapsedTime(&(ms), s__, e__); (ms) /= (float)(reps);         \
        cudaEventDestroy(s__); cudaEventDestroy(e__);                         \
    } while (0)

#define BLK  256          /* 8 warps per block */
#define RANK 16           /* length of each thread's result vector */
#define NNZ  32           /* nonzeros processed per thread */

/* OPTIMIZED -- the per-thread result vector lives in registers.
   No shared memory, no barriers.  The unroll pragmas are what keep acc[] in
   registers instead of spilling it to local memory. */
__global__ void ttm_reg(const float * __restrict__ X, const float * __restrict__ U,
                        float *Y, int n) {
    float acc[RANK];                               /* <-- registers */
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i >= n) return;                            /* n need not divide BLK */

#pragma unroll
    for (int r = 0; r < RANK; ++r) acc[r] = 0.0f;

    for (int k = 0; k < NNZ; ++k) {
        float v = X[(size_t)k * n + i];            /* unchanged */
#pragma unroll
        for (int r = 0; r < RANK; ++r)
            acc[r] += v * U[k * RANK + r];         /* <-- register accumulate */
    }

#pragma unroll
    for (int r = 0; r < RANK; ++r) Y[(size_t)i * RANK + r] = acc[r];
}

int main(void) {
    const int n = 1 << 16;                         /* 65,536 threads of work */
    const size_t xn = (size_t)NNZ * n;             /* 2.1 M floats, 8.4 MB */
    const int blocks = (n + BLK - 1) / BLK;

    float *X = (float *)malloc(xn * sizeof(float));
    float *U = (float *)malloc((size_t)NNZ * RANK * sizeof(float));
    float *Y = (float *)malloc((size_t)n * RANK * sizeof(float));
    for (size_t t = 0; t < xn; ++t) X[t] = (float)((t * 2654435761u) % 17) * 0.25f;
    for (int t = 0; t < NNZ * RANK; ++t) U[t] = (float)(t % 7) * 0.125f;

    float *dX, *dU, *dY;
    CUDA_CHECK(cudaMalloc(&dX, xn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dU, (size_t)NNZ * RANK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, (size_t)n * RANK * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX, X, xn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dU, U, (size_t)NNZ * RANK * sizeof(float), cudaMemcpyHostToDevice));

    printf("Abuse of shared memory removed  [optimized]\n");
    printf("  n=%d, RANK=%d, NNZ/thread=%d, blockDim=%d\n", n, RANK, NNZ, BLK);
    printf("  shared memory: 0 B/block  (was %zu B)\n",
           (size_t)BLK * RANK * sizeof(float));
    printf("  X is %.1f MB; %d register accumulations per thread, 0 SMEM ops\n",
           (double)xn * sizeof(float) / 1e6, NNZ * RANK);

    ttm_reg<<<blocks, BLK>>>(dX, dU, dY, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(Y, dY, (size_t)n * RANK * sizeof(float), cudaMemcpyDeviceToHost));

    double max_rel = 0.0;
    for (int i = 0; i < n; i += 97) {              /* spot-check every 97th row */
        for (int r = 0; r < RANK; ++r) {
            double s = 0.0;
            for (int k = 0; k < NNZ; ++k) s += (double)X[(size_t)k * n + i] * U[k * RANK + r];
            double d = fabs((double)Y[(size_t)i * RANK + r] - s);
            double rr = d / (fabs(s) + 1e-6);
            if (rr > max_rel) max_rel = rr;
        }
    }
    printf("\nCorrectness: %s  (max rel error %.3e)\n",
           max_rel < 1e-4 ? "PASS" : "FAIL", max_rel);

    printf("\nRANK sweep is compile-time here; edit RANK and rebuild.\n");
    printf("Problem-size sweep:\n");
    for (int sub = n / 4; sub <= n; sub *= 2) {
        int sb = (sub + BLK - 1) / BLK;
        float ms = 0.0f;
        TIME_KERNEL(ms, 50, (ttm_reg<<<sb, BLK>>>(dX, dU, dY, n)));
        printf("  %6d threads : %.4f ms\n", sub, ms);
    }

    float ms = 0.0f;
    TIME_KERNEL(ms, 100, (ttm_reg<<<blocks, BLK>>>(dX, dU, dY, n)));
    printf("\nTiming: %.4f ms\n", ms);
    printf("\nCompare with smem_abuse/ttm_naive.cu (same math via shared memory).\n");

    cudaFree(dX); cudaFree(dU); cudaFree(dY);
    free(X); free(U); free(Y);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
Abuse of shared memory removed  [optimized]
  n=65536, RANK=16, NNZ/thread=32, blockDim=256
  shared memory: 0 B/block  (was 16384 B)
  X is 8.4 MB; 512 register accumulations per thread, 0 SMEM ops

Correctness: PASS  (max rel error 0.000e+00)

Problem-size sweep:
   16384 threads : 0.0223 ms
   32768 threads : 0.0391 ms
   65536 threads : 0.0657 ms

Timing: 0.0662 ms      vs ttm_naive.cu 0.0745 ms = 1.13x.

1.13x is a modest win and an honest one: the kernel still has to stream 8.4 MB
of X from memory, and that floor is the same for both versions. The structural
win is the 16 KB of shared memory per block handed back to L1.
*/

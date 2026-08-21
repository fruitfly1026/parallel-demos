/* ============================================================================
   cuThermo pattern: ABUSE OF SHARED MEMORY -- the UNOPTIMIZED version.
   [paper Fig. 5(a); Sec. 6.2, PASTA's spt_TTMRankRBNnzKernelSM / Y_shr]
   ----------------------------------------------------------------------------
   PROVENANCE
       PASTA's kernel cannot be lifted out of its application -- it depends on
       the sparse tensor format and hundreds of lines of setup.  This file is a
       minimal kernel with the SAME SHAPE: a tensor-times-matrix inner loop
       where each thread accumulates a RANK-long result vector, and that vector
       is parked in shared memory (PASTA calls it Y_shr) even though no other
       thread ever reads it.  The pattern is faithful; the timings are this
       benchmark's, not PASTA's 163.56%.

   WHAT THE HEAT MAP LOOKS LIKE
       A line whose address tag lies in the SMEM address space where every
       word's distinct-warp count is 1 and the sector total is 1 as well:

           SMEM tag | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |   sector total = 1
                      ^^^ every word private -- nothing is ever shared

   WHY IT COSTS YOU
       Y_shr[tid][r] is written and read only by thread tid.  Every one of the
       RANK accumulations per nonzero becomes a shared-memory read-modify-write
       -- ~25 cycles of round trip and an LDS/STS instruction pair each -- for
       data a register would hold at 1 cycle.  On top of that the array costs
       BLK*RANK*4 = 16 KB of the 128 KB the SM shares between L1 and SMEM.

   A NOTE ON MEASURING THIS PATTERN
       It is easy to build a version of this demo that shows nothing.  If the
       kernel is DRAM-bandwidth-bound, the SMEM traffic hides completely behind
       the memory system and both versions run at identical speed.  Worse, if
       each thread's scratch is a SINGLE scalar, nvcc simply promotes it to a
       register on its own and the two versions compile to the same code.
       This demo avoids both traps: the working set is small enough to stay
       cache-resident, and the scratch is a RANK-long vector, which is large
       enough that the compiler leaves it in shared memory.

   FIX: see smem_abuse/ttm_opt.cu.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o ttm_naive ttm_naive.cu
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

/* INEFFICIENT -- the per-thread result vector lives in shared memory.
   16 KB/block, and RANK shared-memory read-modify-writes per nonzero. */
__global__ void ttm_smem(const float * __restrict__ X, const float * __restrict__ U,
                         float *Y, int n) {
    __shared__ float Y_shr[BLK][RANK];             /* <-- 16 KB, never shared */
    int tid = threadIdx.x;
    int i   = blockIdx.x * BLK + tid;
    if (i >= n) return;                            /* n need not divide BLK */

    for (int r = 0; r < RANK; ++r) Y_shr[tid][r] = 0.0f;
    __syncthreads();                               /* <-- buys nothing */

    for (int k = 0; k < NNZ; ++k) {
        float v = X[(size_t)k * n + i];            /* coalesced on purpose */
        for (int r = 0; r < RANK; ++r)
            Y_shr[tid][r] += v * U[k * RANK + r];  /* <-- SMEM read-modify-write */
    }
    __syncthreads();                               /* <-- and again */

    for (int r = 0; r < RANK; ++r) Y[(size_t)i * RANK + r] = Y_shr[tid][r];
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

    printf("Abuse of shared memory  [unoptimized]\n");
    printf("  n=%d, RANK=%d, NNZ/thread=%d, blockDim=%d\n", n, RANK, NNZ, BLK);
    printf("  shared memory: %zu B/block  (%d threads x %d floats)\n",
           (size_t)BLK * RANK * sizeof(float), BLK, RANK);
    printf("  X is %.1f MB; %d SMEM read-modify-writes per thread\n",
           (double)xn * sizeof(float) / 1e6, NNZ * RANK);

    ttm_smem<<<blocks, BLK>>>(dX, dU, dY, n);
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
        TIME_KERNEL(ms, 50, (ttm_smem<<<sb, BLK>>>(dX, dU, dY, n)));
        printf("  %6d threads : %.4f ms\n", sub, ms);
    }

    float ms = 0.0f;
    TIME_KERNEL(ms, 100, (ttm_smem<<<blocks, BLK>>>(dX, dU, dY, n)));
    printf("\nTiming: %.4f ms\n", ms);
    printf("\nCompare with smem_abuse/ttm_opt.cu (same math, registers).\n");

    cudaFree(dX); cudaFree(dU); cudaFree(dY);
    free(X); free(U); free(Y);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
Abuse of shared memory  [unoptimized]
  n=65536, RANK=16, NNZ/thread=32, blockDim=256
  shared memory: 16384 B/block  (256 threads x 16 floats)
  X is 8.4 MB; 512 SMEM read-modify-writes per thread

Correctness: PASS  (max rel error 0.000e+00)

Problem-size sweep:
   16384 threads : 0.0249 ms
   32768 threads : 0.0431 ms
   65536 threads : 0.0745 ms

Timing: 0.0745 ms      vs ttm_opt.cu 0.0662 ms = 1.13x slower.
*/

/* ============================================================================
   cuThermo pattern: MEMORY FALSE SHARING -- second example, UNOPTIMIZED.
   [paper Fig. 5(b)]
   ----------------------------------------------------------------------------
   The GEMM pair in this folder (gemm_naive.cu / gemm_opt.cu) shows false
   sharing as it appears in a real library kernel.  THIS pair strips the
   pattern down to the smallest program that produces it, so the heat map can
   be read directly against Fig. 5(b).

   It is also the GPU twin of the CPU demo already in this repo,
   ../../openmp/05_false_sharing.c -- with one important difference, below.

   THE WORKLOAD
       A block of 256 threads = 8 warps.  Each warp reduces its own 32 elements
       with __shfl_down_sync, and lane 0 stores the warp's result:

           partial[blockIdx.x * 8 + wid] = v;

       Those 8 floats are 32 bytes -- exactly one sector -- and each of the 8
       words is written by a DIFFERENT warp:

           Sector | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |   sector total = 8
                    ^ one warp per word        ^ but the sector is 8x hotter

       That is the Fig. 5(b) signature, and the reason it is false sharing
       rather than a hot spot: the sector's temperature is far above any
       individual word's.

   WHY IT COSTS YOU
       Coalescing merges requests only WITHIN a warp.  Eight warps each writing
       one word of the same sector cannot be merged, so the hardware issues 8
       sector transactions to store 32 bytes -- 8x the traffic for the same
       data.

   HOW THIS DIFFERS FROM CPU FALSE SHARING
       On a CPU the cost is coherence: two cores ping-pong an exclusive cache
       line back and forth. On a GPU the warps are on the same SM and there is
       no coherence traffic -- the cost is purely that the 8 writes cannot be
       coalesced into one transaction. Same name, different mechanism, and
       worth saying out loud to students who have just seen the OpenMP version.


   ############################################################################
   #  WHAT THIS PAIR IS AND IS NOT FOR                                        #
   ############################################################################

   This pair exists to make the SIGNATURE unmistakable, not to win time.
   Measured on an RTX A4500, n = 16.7 M:

                              warpsum_naive   warpsum_opt
       global store sectors      524,288         65,536     <- exactly 8x, as
                                                               Fig. 5(b) predicts
       kernel time              0.1426 ms       0.1512 ms   <- 6% SLOWER

   The store transactions collapse by the predicted factor of 8 and the clock
   goes the wrong way.  Both facts are real.  A reduction reads 64 MB and
   writes 2 MB, so stores are ~3% of the traffic; removing 7/8 of a 3% slice
   cannot outrun the __syncthreads() and the 7-of-8 idle warps the fix adds.

   So read this pair for the heat map, and read gemm_naive.cu / gemm_opt.cu in
   the same folder for what false sharing costs when it sits on the critical
   path: there the identical pattern is worth 8.14x of wall-clock time.

   The lesson is the one the whole folder set keeps repeating: a pattern in the
   heat map tells you what the memory system is doing, not whether fixing it
   will pay. Check what the kernel is bound by first.

   FIX: see false_sharing/warpsum_opt.cu.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o warpsum_naive warpsum_naive.cu
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

#define BLK   256                /* 8 warps -- 8 words -- exactly one sector */
#define NWARP (BLK / 32)

__global__ void warpsum_false_sharing(const float * __restrict__ x,
                                      float *partial, int n) {
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    int i   = blockIdx.x * BLK + tid;
    float v = (i < n) ? x[i] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0)
        partial[blockIdx.x * NWARP + wid] = v;  /* <-- 8 warps, 1 sector, 8 txns */
}

int main(void) {
    const int n = 1 << 24;                      /* 16.7 M elements */
    const int blocks = (n + BLK - 1) / BLK;
    const int np = blocks * NWARP;

    float *x   = (float *)malloc((size_t)n * sizeof(float));
    float *p   = (float *)malloc((size_t)np * sizeof(float));
    float *ref = (float *)malloc((size_t)np * sizeof(float));
    for (int i = 0; i < n; ++i) x[i] = (float)(i & 15) * 0.125f;
    for (int w = 0; w < np; ++w) {
        double s = 0.0;
        for (int j = 0; j < 32; ++j) { int i = w * 32 + j; if (i < n) s += x[i]; }
        ref[w] = (float)s;
    }

    float *dx, *dp;
    CUDA_CHECK(cudaMalloc(&dx, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dp, (size_t)np * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, x, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

    printf("Per-warp accumulators, FALSE SHARING  [unoptimized]\n");
    printf("  n=%d, blockDim=%d (%d warps/block), %d partials (%.1f MB)\n",
           n, BLK, NWARP, np, (double)np * sizeof(float) / 1e6);
    printf("  %d floats per block = %zu B = one 32-byte sector, written by %d warps\n",
           NWARP, NWARP * sizeof(float), NWARP);

    CUDA_CHECK(cudaMemset(dp, 0, (size_t)np * sizeof(float)));
    warpsum_false_sharing<<<blocks, BLK>>>(dx, dp, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(p, dp, (size_t)np * sizeof(float), cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (int w = 0; w < np; ++w) {
        double e = fabs((double)p[w] - (double)ref[w]);
        if (e > max_err) max_err = e;
    }
    printf("\nCorrectness: %s  (max error %.3e, all %d partials checked)\n",
           max_err < 1e-3 ? "PASS" : "FAIL", max_err, np);

    printf("\nElement-count sweep:\n");
    for (int sub = n / 4; sub <= n; sub *= 2) {
        int sb = (sub + BLK - 1) / BLK;
        float ms = 0.0f;
        TIME_KERNEL(ms, 50, (warpsum_false_sharing<<<sb, BLK>>>(dx, dp, sub)));
        printf("  %9d elements : %.4f ms   %6.1f GB/s\n",
               sub, ms, (double)sub * sizeof(float) / (ms * 1e-3) / 1e9);
    }

    float ms = 0.0f;
    TIME_KERNEL(ms, 50, (warpsum_false_sharing<<<blocks, BLK>>>(dx, dp, n)));
    printf("\nTiming: %.4f ms\n", ms);
    printf("\nCompare with false_sharing/warpsum_opt.cu (one warp writes the sector).\n");

    cudaFree(dx); cudaFree(dp); free(x); free(p); free(ref);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
Per-warp accumulators, FALSE SHARING  [unoptimized]
  n=16777216, blockDim=256 (8 warps/block), 524288 partials (2.1 MB)

Correctness: PASS  (max error 0.000e+00, all 524288 partials checked)

Element-count sweep:
    4194304 elements : 0.0376 ms    446.7 GB/s
    8388608 elements : 0.0726 ms    462.4 GB/s
   16777216 elements : 0.1425 ms    470.9 GB/s

Timing: 0.1426 ms

Nsight Compute, l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum:
  524,288 store sectors  (warpsum_opt.cu: 65,536 -- exactly 8x fewer)
*/

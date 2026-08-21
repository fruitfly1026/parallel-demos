/* ============================================================================
   cuThermo pattern: MEMORY FALSE SHARING -- second example, OPTIMIZED.
   [paper Fig. 5(b)]
   ----------------------------------------------------------------------------
   ############################################################################
   #  WHAT CHANGED vs false_sharing/warpsum_naive.cu                          #
   ############################################################################

   THE WARP-LEVEL REDUCTION IS UNTOUCHED.  Only the store changes: instead of
   8 warps each writing one word of a sector, the 8 results are parked in
   shared memory and ONE warp writes all 8 words.

       naive:                                  opt:
           if (lane == 0)                          __shared__ float s[NWARP];
             partial[blockIdx.x*NWARP + wid] = v;  if (lane == 0) s[wid] = v;
                                                   __syncthreads();
                                                   if (tid < NWARP)
                                                     partial[blockIdx.x*NWARP + tid]
                                                       = s[tid];

   WHY THAT IS THE RIGHT FIX
       The instinct carried over from CPU false sharing is to PAD -- give each
       warp its own cache line. That is the wrong move here. Padding to one
       sector per warp still costs 8 sector transactions, it just wastes 8x the
       memory as well. The GPU problem is not coherence, it is that coalescing
       only works *within* a warp. So the fix is to make the words of a sector
       be written by the SAME warp, which is what routing them through shared
       memory and letting warp 0 do the store achieves: 8 words, one warp, one
       coalesced transaction.

       This is the same principle as gemm_opt.cu in this folder -- there it was
       achieved by remapping thread indices, here by re-routing the store.

   THE COST: NWARP floats of shared memory (32 bytes) and one __syncthreads().

   WHAT THE HEAT MAP SHOWS
       naive  Sector | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |   total = 8
       opt    Sector | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |   total = 1

       The per-word temperatures are unchanged -- each word still has exactly
       one writer. Only the SECTOR total moves, from 8 to 1. That is precisely
       the distinction cuThermo's word-vs-sector granularity exists to make,
       and a word-level-only profiler would show these two kernels as identical.


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

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o warpsum_opt warpsum_opt.cu
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

__global__ void warpsum_one_writer(const float * __restrict__ x,
                                   float *partial, int n) {
    __shared__ float s[NWARP];                  /* 32 bytes */
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    int i   = blockIdx.x * BLK + tid;
    float v = (i < n) ? x[i] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)      /* unchanged */
        v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) s[wid] = v;
    __syncthreads();
    if (tid < NWARP)
        partial[blockIdx.x * NWARP + tid] = s[tid];  /* <-- warp 0 writes all 8 */
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

    printf("Per-warp accumulators, one writer per sector  [optimized]\n");
    printf("  n=%d, blockDim=%d (%d warps/block), %d partials (%.1f MB)\n",
           n, BLK, NWARP, np, (double)np * sizeof(float) / 1e6);
    printf("  %d floats per block = %zu B = one 32-byte sector, written by 1 warp\n",
           NWARP, NWARP * sizeof(float));

    CUDA_CHECK(cudaMemset(dp, 0, (size_t)np * sizeof(float)));
    warpsum_one_writer<<<blocks, BLK>>>(dx, dp, n);
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
        TIME_KERNEL(ms, 50, (warpsum_one_writer<<<sb, BLK>>>(dx, dp, sub)));
        printf("  %9d elements : %.4f ms   %6.1f GB/s\n",
               sub, ms, (double)sub * sizeof(float) / (ms * 1e-3) / 1e9);
    }

    float ms = 0.0f;
    TIME_KERNEL(ms, 50, (warpsum_one_writer<<<blocks, BLK>>>(dx, dp, n)));
    printf("\nTiming: %.4f ms\n", ms);
    printf("\nCompare with false_sharing/warpsum_naive.cu (8 warps write the sector).\n");

    cudaFree(dx); cudaFree(dp); free(x); free(p); free(ref);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
Per-warp accumulators, one writer per sector  [optimized]
  n=16777216, blockDim=256 (8 warps/block), 524288 partials (2.1 MB)

Correctness: PASS  (max error 0.000e+00, all 524288 partials checked)

Element-count sweep:
    4194304 elements : 0.0398 ms    422.1 GB/s
    8388608 elements : 0.0769 ms    436.4 GB/s
   16777216 elements : 0.1512 ms    443.9 GB/s

Timing: 0.1512 ms

Nsight Compute, l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum:
  65,536 store sectors  (warpsum_naive.cu: 524,288 -- exactly 8x more)

The sector count is the result; the 6% time regression is real and explained
in the header. See gemm_naive.cu / gemm_opt.cu for false sharing that costs
8.14x of actual run time.
*/

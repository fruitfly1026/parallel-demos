/* ============================================================================
   WARP (THREAD) DIVERGENCE -- the OPTIMIZED version.
   ----------------------------------------------------------------------------
   SOURCE: `reduce1` and `reduce2` from NVIDIA's CUDA sample
   (cuda-samples, cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu),
   steps 2 and 3 of Mark Harris's "Optimizing Parallel Reduction in CUDA".
   Mechanical edits only: cg::sync(cta) -> __syncthreads(), templated
   SharedMemory<T>() -> a plain float array.

   ############################################################################
   #  WHAT CHANGED vs thread_divergence/thread_divergence_naive.cu            #
   ############################################################################

   THE LOOP DIRECTION AND THE BRANCH CONDITION.  Everything else -- the shared
   tile, the load, the barriers, the final store -- is identical.

       naive (reduce0):
           for (unsigned int s = 1; s < blockDim.x; s *= 2) {
               if ((tid % (2 * s)) == 0)              // scattered active lanes
                   sdata[tid] += sdata[tid + s];
               __syncthreads();
           }

       opt (reduce2):
           for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
               if (tid < s)                           // contiguous active lanes
                   sdata[tid] += sdata[tid + s];
               __syncthreads();
           }

   WHY THAT REMOVES THE DIVERGENCE
       `tid < s` is true for a CONTIGUOUS prefix of threads, so a warp is
       either entirely active or entirely inactive -- and an entirely inactive
       warp is simply not scheduled.  Compare the two:

           step   naive: active lanes           opt: active warps
           s=1    16 of 32 in EVERY warp        warps 0..3 fully active
           s=2     8 of 32 in EVERY warp        warps 0..1 fully active
           s=4     4 of 32 in EVERY warp        warp 0 fully active
           s=8     2 of 32 in EVERY warp        warp 0, 16 of 32 lanes
           s=16    1 of 32 in EVERY warp        warp 0,  8 of 32 lanes

       The naive version keeps all 8 warps busy doing almost nothing.  The
       optimized one retires warps as the reduction narrows; only the last few
       steps, inside a single warp, are partially masked -- and that is
       unavoidable in a tree reduction.

       The slow integer `%` goes away at the same time: `tid < s` needs no
       division at all.

   IT ALSO FIXES BANK CONFLICTS, WHICH IS A SEPARATE WIN
       reduce2 is called "sequential addressing" because sdata[tid] and
       sdata[tid+s] are both contiguous across lanes, so the shared-memory
       accesses are conflict-free.  reduce0's stride-2s access pattern is not.
       To keep the two effects apart, this file ALSO runs `reduce1`, the
       intermediate step from the same NVIDIA sample:

           for (unsigned int s = 1; s < blockDim.x; s *= 2) {
               int index = 2 * s * tid;
               if (index < blockDim.x) sdata[index] += sdata[index + s];
               __syncthreads();
           }

       reduce1 fixes ONLY the divergence -- active lanes are contiguous again,
       but the stride-2s addressing still causes bank conflicts.  So:

           reduce0 -> reduce1   = the divergence fix alone
           reduce1 -> reduce2   = the bank-conflict fix alone   (see bank_conflict/)

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o thread_divergence_opt thread_divergence_opt.cu
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

#define BLK 256

/* reduce1: divergence removed, bank conflicts remain (the intermediate step). */
__global__ void reduce1(const float *g_idata, float *g_odata, unsigned int n) {
    __shared__ float sdata[BLK];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;                   /* contiguous active lanes */
        if (index < blockDim.x) sdata[index] += sdata[index + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/* reduce2: sequential addressing -- no divergence, no bank conflicts. */
__global__ void reduce2(const float *g_idata, float *g_odata, unsigned int n) {
    __shared__ float sdata[BLK];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s]; /* whole warps, or none */
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main(void) {
    const unsigned int n = 1u << 24;               /* 16.7 M elements */
    const int blocks = (int)((n + BLK - 1) / BLK);

    float *h = (float *)malloc((size_t)n * sizeof(float));
    float *p = (float *)malloc((size_t)blocks * sizeof(float));
    double ref = 0.0;
    for (unsigned int i = 0; i < n; ++i) { h[i] = (float)(i % 7) * 0.25f; ref += h[i]; }

    float *din, *dpart;
    CUDA_CHECK(cudaMalloc(&din, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dpart, (size_t)blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(din, h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

    printf("Parallel reduction, SEQUENTIAL addressing (reduce2)  [optimized]\n");
    printf("  n=%u, blockDim=%d, %d blocks\n", n, BLK, blocks);
    printf("  active lanes are a contiguous prefix -- whole warps retire early\n");

    /* ---- correctness for both kernels ---- */
    printf("\nCorrectness:\n");
    for (int v = 0; v < 2; ++v) {
        CUDA_CHECK(cudaMemset(dpart, 0, (size_t)blocks * sizeof(float)));
        if (v == 0) reduce1<<<blocks, BLK>>>(din, dpart, n);
        else        reduce2<<<blocks, BLK>>>(din, dpart, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(p, dpart, (size_t)blocks * sizeof(float), cudaMemcpyDeviceToHost));
        double got = 0.0;
        for (int b = 0; b < blocks; ++b) got += p[b];
        double rel = fabs(got - ref) / (fabs(ref) + 1e-9);
        printf("  %-28s %s  (rel error %.3e)\n",
               v == 0 ? "reduce1 (divergence fixed)" : "reduce2 (also conflict-free)",
               rel < 1e-5 ? "PASS" : "FAIL", rel);
    }

    /* ---- parameter sweep ---- */
    printf("\nElement-count sweep (reduce2):\n");
    for (unsigned int sub = n / 4; sub <= n; sub *= 2) {
        int sb = (int)((sub + BLK - 1) / BLK);
        float ms = 0.0f;
        TIME_KERNEL(ms, 50, (reduce2<<<sb, BLK>>>(din, dpart, sub)));
        printf("  %9u elements : %.4f ms   %6.1f GB/s\n",
               sub, ms, (double)sub * sizeof(float) / (ms * 1e-3) / 1e9);
    }

    float t1 = 0.0f, t2 = 0.0f;
    TIME_KERNEL(t1, 50, (reduce1<<<blocks, BLK>>>(din, dpart, n)));
    TIME_KERNEL(t2, 50, (reduce2<<<blocks, BLK>>>(din, dpart, n)));
    printf("\nTiming, isolating the two effects:\n");
    printf("  reduce1 (divergence fixed, conflicts remain) : %.4f ms\n", t1);
    printf("  reduce2 (also conflict-free)                 : %.4f ms\n", t2);
    printf("\nCompare with thread_divergence_naive.cu (reduce0) for the divergence cost.\n");

    cudaFree(din); cudaFree(dpart); free(h); free(p);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
Parallel reduction, SEQUENTIAL addressing (reduce2)  [optimized]
  n=16777216, blockDim=256, 65536 blocks

Correctness:
  reduce1 (divergence fixed)   PASS  (rel error 0.000e+00)
  reduce2 (also conflict-free) PASS  (rel error 0.000e+00)

Element-count sweep (reduce2):
    4194304 elements : 0.0889 ms    188.7 GB/s
    8388608 elements : 0.1750 ms    191.8 GB/s
   16777216 elements : 0.3473 ms    193.3 GB/s

Timing, isolating the two effects:
  reduce0 (naive file)                         : 0.5030 ms
  reduce1 (divergence fixed, conflicts remain) : 0.3615 ms   <- 1.39x, divergence
  reduce2 (also conflict-free)                 : 0.3472 ms   <- 1.04x, conflicts
                                                                 1.45x overall

Divergence is the larger of the two effects here, which is why this folder
uses reduce0 -> reduce2 while bank_conflict/ uses the transpose instead.
*/

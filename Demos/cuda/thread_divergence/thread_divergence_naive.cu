/* ============================================================================
   WARP (THREAD) DIVERGENCE -- the UNOPTIMIZED version.
   ----------------------------------------------------------------------------
   WHAT DIVERGENCE IS
       A warp's 32 lanes share one instruction pointer.  When a branch sends
       some lanes one way and the rest another, the hardware executes BOTH
       paths in sequence and masks off the inactive lanes.  A warp in which
       only 1 of 32 lanes is doing useful work still costs a full instruction
       issue -- you are paying for 32 lanes and using one.

   THE WORKLOAD -- parallel reduction, "interleaved addressing"
       This is `reduce0` from NVIDIA's own CUDA sample
       (cuda-samples, cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu)
       and step 1 of Mark Harris's "Optimizing Parallel Reduction in CUDA".
       The only edits are mechanical, so the file builds standalone:
       cg::sync(cta) -> __syncthreads(), and the templated SharedMemory<T>()
       helper -> a plain float array.

   WHERE THE DIVERGENCE IS
           for (unsigned int s = 1; s < blockDim.x; s *= 2) {
               if ((tid % (2 * s)) == 0)          // <-- divergent
                   sdata[tid] += sdata[tid + s];
               __syncthreads();
           }

       The active lanes are the ones whose tid is a multiple of 2*s, and they
       are SPREAD OUT across the warp:

           s =  1 : lanes 0,2,4,...,30 active  -> 16 of 32   (50%)
           s =  2 : lanes 0,4,8,...,28 active  ->  8 of 32   (25%)
           s =  4 : lanes 0,8,16,24    active  ->  4 of 32   (12.5%)
           s =  8 : lanes 0,16         active  ->  2 of 32   (6.25%)
           s = 16 : lane  0            active  ->  1 of 32   (3.1%)

       Every warp in the block runs every iteration, because every warp
       contains at least one active lane.  By the last steps the machine is
       issuing full-width instructions to do one addition.

       The `%` is a second, separate problem -- integer modulo is slow, and
       NVIDIA's own comment in the sample says so ("modulo arithmetic is
       slow!").  The fix in thread_divergence_opt.cu removes both at once.

   FIX: see thread_divergence/thread_divergence_opt.cu.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o thread_divergence_naive thread_divergence_naive.cu
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

/* reduce0 from the NVIDIA reduction sample: interleaved addressing with a
   divergent modulo branch. */
__global__ void reduce0(const float *g_idata, float *g_odata, unsigned int n) {
    __shared__ float sdata[BLK];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        /* modulo arithmetic is slow, and the active lanes are scattered */
        if ((tid % (2 * s)) == 0) {
            sdata[tid] += sdata[tid + s];
        }
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

    printf("Parallel reduction, INTERLEAVED addressing (reduce0)  [unoptimized]\n");
    printf("  n=%u, blockDim=%d, %d blocks\n", n, BLK, blocks);
    printf("  active lanes per warp by step: 16, 8, 4, 2, 1 of 32\n");

    /* ---- correctness: block partials summed on the host ---- */
    reduce0<<<blocks, BLK>>>(din, dpart, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(p, dpart, (size_t)blocks * sizeof(float), cudaMemcpyDeviceToHost));
    double got = 0.0;
    for (int b = 0; b < blocks; ++b) got += p[b];
    double rel = fabs(got - ref) / (fabs(ref) + 1e-9);
    printf("\nCorrectness: %s  (got %.6e, expected %.6e, rel error %.3e)\n",
           rel < 1e-5 ? "PASS" : "FAIL", got, ref, rel);

    /* ---- parameter sweep ---- */
    printf("\nblockDim sweep:\n");
    for (int t : {128, 256, 1024}) {
        (void)t;   /* sdata is sized by BLK at compile time; see note below */
    }
    printf("  (sdata is sized by BLK at compile time -- edit BLK and rebuild\n"
           "   to vary the block size; the divergence pattern is the same.)\n");
    printf("\nElement-count sweep:\n");
    for (unsigned int sub = n / 4; sub <= n; sub *= 2) {
        int sb = (int)((sub + BLK - 1) / BLK);
        float ms = 0.0f;
        TIME_KERNEL(ms, 50, (reduce0<<<sb, BLK>>>(din, dpart, sub)));
        printf("  %9u elements : %.4f ms   %6.1f GB/s\n",
               sub, ms, (double)sub * sizeof(float) / (ms * 1e-3) / 1e9);
    }

    float ms = 0.0f;
    TIME_KERNEL(ms, 50, (reduce0<<<blocks, BLK>>>(din, dpart, n)));
    printf("\nTiming: %.4f ms   %.1f GB/s\n",
           ms, (double)n * sizeof(float) / (ms * 1e-3) / 1e9);
    printf("\nCompare with thread_divergence/thread_divergence_opt.cu (reduce2).\n");

    cudaFree(din); cudaFree(dpart); free(h); free(p);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
Parallel reduction, INTERLEAVED addressing (reduce0)  [unoptimized]
  n=16777216, blockDim=256, 65536 blocks

Correctness: PASS  (got 1.258291e+07, expected 1.258291e+07, rel error 0.000e+00)

Element-count sweep:
    4194304 elements : 0.1282 ms    130.9 GB/s
    8388608 elements : 0.2531 ms    132.6 GB/s
   16777216 elements : 0.5029 ms    133.4 GB/s

Timing: 0.5030 ms   133.4 GB/s

vs thread_divergence_opt.cu: 0.5030 -> 0.3472 ms = 1.45x total, of which
1.39x is the divergence fix alone (reduce1) and 1.04x the bank-conflict fix.
*/

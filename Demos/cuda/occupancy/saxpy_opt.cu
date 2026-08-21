/* ============================================================================
   OCCUPANCY -- the OPTIMIZED version.  SAXPY (y = a*x + y) with the grid sized
   to saturate every SM.
   Split out of the repo's 03_saxpy.cu; the unoptimized one is
   occupancy/saxpy_naive.cu.
   ----------------------------------------------------------------------------
   ############################################################################
   #  WHAT CHANGED vs occupancy/saxpy_naive.cu                                #
   ############################################################################

   NOT ONE LINE OF THE KERNEL.  `saxpy` is character-for-character identical in
   both files -- the grid-stride loop was already correct for any launch shape.
   The entire fix is the LAUNCH CONFIGURATION on the host side:

       saxpy_naive.cu:
           saxpy<<<1, threads>>>(n, a, x, y);              // 1 block, 1 SM busy

       saxpy_opt.cu:
           int sm = 0;
           cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
           int blocks_full = sm * 32;                      // query the device...
           saxpy<<<blocks_full, threads>>>(n, a, x, y);    // ...and fill it

   The `* 32` gives each SM many resident blocks so the scheduler can hide
   memory latency by swapping between warps; 1x the SM count is enough to touch
   every SM but not enough to hide latency, which the sweep below shows.

   WHY IT MATTERS FOR PROFILING
       This is the one folder here whose problem a MEMORY heat map cannot see.
       Both versions have identical, perfectly coalesced access patterns, so
       cuThermo reports nothing wrong in either -- and it is right to. The
       lesson is to check occupancy (Nsight Compute's launch statistics)
       BEFORE reading a memory heat map: a kernel that is 57x slow because the
       GPU is empty has no memory problem to find.

     (a) CORRECTNESS CHECK  — GPU result vs a CPU reference, PASS/FAIL.
     (b) PARAMETER VARYING  — grid = 1x / 8x / 32x the SM count.
     (c) THE FIX            — see above.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o saxpy_opt saxpy_opt.cu
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

/* Identical to the kernel in saxpy_naive.cu -- nothing here needed fixing. */
__global__ void saxpy(int n, float a, const float *x, float *y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x)        /* grid-stride loop */
        y[i] = a * x[i] + y[i];
}

static float time_saxpy(int blocks, int threads, int n, float a,
                        const float *x, float *y, int reps) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    saxpy<<<blocks, threads>>>(n, a, x, y);      /* warm-up */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int r = 0; r < reps; r++) saxpy<<<blocks, threads>>>(n, a, x, y);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / (float)reps;
}

int main(void) {
    int n = 1 << 26;                             /* 67 M elements */
    size_t bytes = (size_t)n * sizeof(float);
    const float a = 3.0f;
    float *x, *y;
    CUDA_CHECK(cudaMallocManaged(&x, bytes));    /* unified memory */
    CUDA_CHECK(cudaMallocManaged(&y, bytes));
    for (int i = 0; i < n; i++) { x[i] = 1.0f; y[i] = 2.0f; }

    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int threads = 256;
    int blocks_full = sm * 32;                   /* plenty to fill the GPU */

    printf("SAXPY, grid sized to the device: %d SMs x 32 = %d blocks  [optimized]\n",
           sm, blocks_full);

    /* ---- (a) CORRECTNESS ---- */
    saxpy<<<blocks_full, threads>>>(n, a, x, y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double ref = (double)a * 1.0 + 2.0;      /* y started at 2, x=1 */
        double e = fabs((double)y[i] - ref);
        if (e > max_err) max_err = e;
    }
    printf("\nCorrectness (n=%d): %s  (max error %.3e, expect y=5)\n",
           n, max_err < 1e-4 ? "PASS" : "FAIL", max_err);
    for (int i = 0; i < n; i++) y[i] = 2.0f;     /* reset for timing */

    /* ---- (b) PARAMETER VARYING ---- */
    printf("\nGrid-size sweep (%d SMs, blockDim %d):\n", sm, threads);
    for (int mult : {1, 8, 32}) {
        int blocks = sm * mult;
        float ms = time_saxpy(blocks, threads, n, a, x, y, 50);
        double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;   /* 2 reads + 1 write */
        printf("  %2dx SMs = %5d blocks : %.3f ms   %.1f GB/s\n",
               mult, blocks, ms, gbps);
    }
    printf("\nCompare with occupancy/saxpy_naive.cu (1 block, 1 SM busy).\n");

    cudaFree(x); cudaFree(y);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
SAXPY, grid sized to the device: 56 SMs x 32 = 1792 blocks  [optimized]

Correctness (n=67108864): PASS  (max error 0.000e+00, expect y=5)

Grid-size sweep (56 SMs, blockDim 256):
   1x SMs =    56 blocks : 2.099 ms   383.7 GB/s
   8x SMs =   448 blocks : 1.451 ms   554.9 GB/s
  32x SMs =  1792 blocks : 1.437 ms   560.5 GB/s

Note 1x SMs (56 blocks) already touches every SM but still leaves 1.5x on the
table -- one block per SM cannot hide memory latency. vs saxpy_naive.cu:
85.588 -> 1.437 ms = 59.6x.
*/

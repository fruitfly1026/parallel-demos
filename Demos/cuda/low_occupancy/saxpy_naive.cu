/* ============================================================================
   OCCUPANCY -- the UNOPTIMIZED version.  SAXPY (y = a*x + y) launched on a
   single thread block, so 55 of the A4500's 56 SMs sit idle.
   Split out of the repo's 03_saxpy.cu; the fix is low_occupancy/saxpy_opt.cu.
   ----------------------------------------------------------------------------
   THE INEFFICIENCY
       The kernel itself is fine: a grid-stride loop is correct for ANY launch
       configuration.  That is exactly what makes this failure mode easy to
       miss -- the code produces the right answer, passes every test, and is
       simply 57x slower than it should be.

       With <<<1, 256>>> the whole 67 M-element array is walked by one block on
       one SM.  There is no memory-access pattern problem here at all: each
       warp's loads are perfectly coalesced.  The GPU is idle, not confused.
       This is why cuThermo's heat map will look CLEAN for this kernel -- a
       memory heat map cannot see an occupancy problem, and that contrast is
       the point of including it next to the memory-pattern folders.

     (a) CORRECTNESS CHECK  — GPU result vs a CPU reference, PASS/FAIL.
     (b) PARAMETER VARYING  — 1 / 2 / 4 blocks, all far too few.
     (c) THE INEFFICIENCY   — see above; FIX in low_occupancy/saxpy_opt.cu.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o saxpy_naive saxpy_naive.cu
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

/* Grid-stride loop: correct for ANY number of blocks. With too few blocks the
   loop still finishes the work, it just leaves most SMs idle. */
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

    printf("SAXPY, ONE thread block -- %d of %d SMs idle  [unoptimized]\n",
           sm - 1, sm);

    /* ---- (a) CORRECTNESS ---- */
    saxpy<<<1, threads>>>(n, a, x, y);
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

    /* ---- (b) PARAMETER VARYING: still nowhere near enough blocks ---- */
    printf("\nUnder-filled grid sweep (%d SMs, blockDim %d):\n", sm, threads);
    for (int blocks : {1, 2, 4}) {
        float ms = time_saxpy(blocks, threads, n, a, x, y, 5);
        double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;   /* 2 reads + 1 write */
        printf("  %2d block(s) : %8.3f ms   %6.1f GB/s   (%d SMs busy)\n",
               blocks, ms, gbps, blocks);
    }
    printf("\nCompare with low_occupancy/saxpy_opt.cu (grid sized to fill all %d SMs).\n", sm);

    cudaFree(x); cudaFree(y);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
SAXPY, ONE thread block -- 55 of 56 SMs idle  [unoptimized]

Correctness (n=67108864): PASS  (max error 0.000e+00, expect y=5)

Under-filled grid sweep (56 SMs, blockDim 256):
   1 block(s) :   85.588 ms      9.4 GB/s   (1 SMs busy)
   2 block(s) :   42.146 ms     19.1 GB/s   (2 SMs busy)
   4 block(s) :   21.341 ms     37.7 GB/s   (4 SMs busy)

Time halves each time the block count doubles -- the work is perfectly
parallel, there is simply no one to do it. vs saxpy_opt.cu: 85.588 / 1.437
= 59.6x slower.
*/

/* SAXPY (y = a*x + y) with a grid-stride loop (handles any n with any launch
   config) and unified memory (no explicit memcpy). Extended to the professor's
   spec:
     (a) CORRECTNESS CHECK  — GPU result vs a CPU reference, PASS/FAIL + max err.
     (b) PARAMETER VARYING  — grid sized to 1x / 8x / 32x the SM count, showing
                              how occupancy changes the run time.
     (c) INEFFICIENT vs OPTIMIZED — a tiny 1-block grid (almost the whole GPU
                              idle) vs a grid sized to fill every SM, timed with
                              cudaEvents.

   Build: nvcc -O3 -arch=sm_86 -o 03_saxpy 03_saxpy.cu                        */
#include <cstdio>
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
                        const float *x, float *y) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    saxpy<<<blocks, threads>>>(n, a, x, y);      /* warm-up */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int r = 0; r < 50; r++) saxpy<<<blocks, threads>>>(n, a, x, y);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / 50.0f;
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

    /* ---- (a) CORRECTNESS: one well-sized launch, compare to CPU. ---- */
    int blocks_full = sm * 32;                   /* plenty to fill the GPU */
    saxpy<<<blocks_full, threads>>>(n, a, x, y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double ref = (double)a * 1.0 + 2.0;      /* y started at 2, x=1 */
        double e = fabs((double)y[i] - ref);
        if (e > max_err) max_err = e;
    }
    printf("Correctness (n=%d): %s  (max error %.3e, expect y=5)\n",
           n, max_err < 1e-4 ? "PASS" : "FAIL", max_err);
    /* Reset y so the timing runs below all start from the same state. */
    for (int i = 0; i < n; i++) y[i] = 2.0f;

    /* ---- (b) PARAMETER VARYING: grid = 1x / 8x / 32x the SM count. ---- */
    printf("\nGrid-size sweep (%d SMs, blockDim %d):\n", sm, threads);
    for (int mult : {1, 8, 32}) {
        int blocks = sm * mult;
        float ms = time_saxpy(blocks, threads, n, a, x, y);
        double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;   /* 2 reads + 1 write */
        printf("  %2dx SMs = %5d blocks : %.3f ms   %.1f GB/s\n",
               mult, blocks, ms, gbps);
    }

    /* ---- (c) INEFFICIENT vs OPTIMIZED. ----
       INEFFICIENT: a single block. Only one SM does any work; the other SMs
       sit idle, so occupancy is ~1/SM and the grid-stride loop is serialized
       across the whole array on one multiprocessor.
       FIX: size the grid to sm*32 blocks so every SM is saturated. */
    float ms_bad  = time_saxpy(1,           threads, n, a, x, y);
    float ms_good = time_saxpy(blocks_full, threads, n, a, x, y);
    printf("\nOccupancy (blockDim %d):\n", threads);
    printf("  1 block (1 SM busy)     : %.3f ms\n", ms_bad);
    printf("  %d blocks (all SMs busy): %.3f ms\n", blocks_full, ms_good);
    printf("  speedup (bad/good)      : %.2fx\n", ms_bad / ms_good);

    cudaFree(x); cudaFree(y);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), Xeon w7-2495X 48c:
Correctness (n=67108864): PASS  (max error 0.000e+00, expect y=5)

Grid-size sweep (56 SMs, blockDim 256):
   1x SMs =    56 blocks : 2.098 ms   383.8 GB/s
   8x SMs =   448 blocks : 1.452 ms   554.7 GB/s
  32x SMs =  1792 blocks : 1.437 ms   560.3 GB/s

Occupancy (blockDim 256):
  1 block (1 SM busy)     : 81.856 ms
  1792 blocks (all SMs busy): 1.436 ms
  speedup (bad/good)      : 57.00x
*/

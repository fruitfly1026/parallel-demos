/* ============================================================================
   SHARED-MEMORY BANK CONFLICTS -- the OPTIMIZED version.
   ----------------------------------------------------------------------------
   ############################################################################
   #  WHAT CHANGED vs bank_conflict/transpose_naive.cu                    #
   ############################################################################

   ONE CHARACTER.  The shared tile gains a padding column:

       naive: __shared__ float tile[TILE][TILE];       // row stride 32 floats
       opt  : __shared__ float tile[TILE][TILE + 1];   // row stride 33 floats

   Every index expression, both loops, the launch configuration and all global
   memory traffic are untouched.  The padding column is never written and never
   read -- it exists only to change the row stride.

   WHY ONE CHARACTER IS ENOUGH
       Shared memory has 32 banks; a float at byte address a is in bank
       (a/4) % 32.  The column read tile[threadIdx.x][c] walks consecutive
       ROWS, so the bank it lands in depends on the row stride:

           stride 32: bank = (r*32 + c) % 32 = c          for every r
                      -> all 32 lanes on bank c -> 32-way conflict
           stride 33: bank = (r*33 + c) % 32 = (r + c) % 32
                      -> lane r on bank (r+c) % 32 -> all 32 banks, no conflict

       Because 33 is coprime with 32, stepping one row also steps one bank, so
       the 32 lanes fan out across all 32 banks exactly once.

   THE COST: TILE extra floats of shared memory per block -- 32 * 4 = 128 bytes
   on a 4 KB tile, i.e. 3%.

   This is NVIDIA's `transposeNoBankConflicts`, from the CUDA Technical Blog
   post "An Efficient Matrix Transpose in CUDA C/C++".

   HOW TO SEE IT IN A PROFILER
       Bank conflicts are a SHARED-memory effect, so they do not appear in a
       global-memory heat map at all.  In Nsight Compute the metric is
           l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
       which should drop to ~0 here while staying large in the naive version.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o transpose_opt transpose_opt.cu
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

#define TILE       32
#define BLOCK_ROWS 8

__global__ void transpose_padded(float *odata, const float * __restrict__ idata,
                                 int width) {
    __shared__ float tile[TILE][TILE + 1];      /* <-- stride 33: conflict-free */

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(size_t)(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;        /* transposed block offset */
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        odata[(size_t)(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
        /*                                   ^^^ same expression, now spread
                                                 across all 32 banks */
}

int main(void) {
    const int width = 2048;                     /* 2048 x 2048 floats = 16 MB */
    /* The grid is width/TILE blocks each way, so a width that TILE does not
       divide would leave the last rows and columns untransposed. Checked here
       so the constraint fails loudly instead of silently. */
    if (width % TILE != 0) {
        fprintf(stderr, "width (%d) must be a multiple of TILE (%d)\n", width, TILE);
        return 1;
    }
    const size_t n = (size_t)width * width;
    const size_t nb = n * sizeof(float);

    float *h = (float *)malloc(nb), *g = (float *)malloc(nb);
    for (size_t t = 0; t < n; ++t) h[t] = (float)(t % 1000) * 0.5f;

    float *din, *dout;
    CUDA_CHECK(cudaMalloc(&din, nb));
    CUDA_CHECK(cudaMalloc(&dout, nb));
    CUDA_CHECK(cudaMemcpy(din, h, nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dout, 0, nb));

    dim3 blk(TILE, BLOCK_ROWS), grd(width / TILE, width / TILE);

    printf("Shared-memory transpose, bank conflicts REMOVED  [optimized]\n");
    printf("  %dx%d floats (%.1f MB), TILE=%d, blockDim %dx%d\n",
           width, width, (double)nb / 1e6, TILE, TILE, BLOCK_ROWS);
    printf("  shared tile: float[%d][%d] -> row stride %d floats -> steps %d bank(s)\n",
           TILE, TILE + 1, TILE + 1, (TILE + 1) % 32);
    printf("  extra shared memory: %zu B/block (%.1f%%)\n",
           TILE * sizeof(float), 100.0 / TILE);

    /* ---- correctness ---- */
    transpose_padded<<<grd, blk>>>(dout, din, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(g, dout, nb, cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (int r = 0; r < width; ++r)
        for (int c = 0; c < width; ++c) {
            double e = fabs((double)g[(size_t)r * width + c] - (double)h[(size_t)c * width + r]);
            if (e > max_err) max_err = e;
        }
    printf("\nCorrectness: %s  (max error %.3e)\n",
           max_err < 1e-6 ? "PASS" : "FAIL", max_err);

    /* ---- parameter sweep ---- */
    printf("\nMatrix-size sweep:\n");
    for (int w : {512, 1024, 2048}) {
        size_t wn = (size_t)w * w, wb = wn * sizeof(float);
        float *ai, *ao;
        CUDA_CHECK(cudaMalloc(&ai, wb)); CUDA_CHECK(cudaMalloc(&ao, wb));
        CUDA_CHECK(cudaMemset(ai, 0, wb));
        dim3 g2(w / TILE, w / TILE);
        float ms = 0.0f;
        TIME_KERNEL(ms, 50, (transpose_padded<<<g2, blk>>>(ao, ai, w)));
        printf("  %4dx%-4d : %.4f ms   %6.1f GB/s\n",
               w, w, ms, 2.0 * (double)wb / (ms * 1e-3) / 1e9);
        cudaFree(ai); cudaFree(ao);
    }

    float ms = 0.0f;
    TIME_KERNEL(ms, 50, (transpose_padded<<<grd, blk>>>(dout, din, width)));
    printf("\nTiming (%dx%d): %.4f ms   %.1f GB/s\n",
           width, width, ms, 2.0 * (double)nb / (ms * 1e-3) / 1e9);
    printf("\nCompare with bank_conflict/transpose_naive.cu (unpadded tile).\n");

    cudaFree(din); cudaFree(dout); free(h); free(g);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
Shared-memory transpose, bank conflicts REMOVED  [optimized]
  2048x2048 floats (16.8 MB), TILE=32, blockDim 32x8
  shared tile: float[32][33] -> row stride 33 floats -> steps 1 bank(s)
  extra shared memory: 128 B/block (3.1%)

Correctness: PASS  (max error 0.000e+00)

Matrix-size sweep:
   512x512  : 0.0031 ms    668.3 GB/s     vs naive 0.0058 ms = 1.87x
  1024x1024 : 0.0163 ms    513.3 GB/s     vs naive 0.0196 ms = 1.20x
  2048x2048 : 0.0615 ms    546.0 GB/s     vs naive 0.0661 ms = 1.07x

Nsight Compute, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum:
  every size : 0 conflicts   (naive: 253,952 at 512, 4,063,232 at 2048)

The conflict count goes to exactly zero at every size, but the speedup shrinks
from 1.87x to 1.07x as the matrix grows. Bank conflicts are an L1/shared-memory
throughput problem, so they only surface while the kernel is NOT yet bound by
DRAM bandwidth. Same lesson as misalignment/ -- know what your kernel is bound
by before you decide a pattern is worth fixing.
*/

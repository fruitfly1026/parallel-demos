/* ============================================================================
   SHARED-MEMORY BANK CONFLICTS -- the UNOPTIMIZED version.
   ----------------------------------------------------------------------------
   WHAT A BANK CONFLICT IS
       Shared memory is split into 32 banks of 4 bytes.  A float at byte
       address a lives in bank (a/4) % 32.  A warp's 32 lanes can be served in
       ONE cycle only if they hit 32 different banks (or all read the same
       address, which broadcasts).  If k lanes hit the same bank at different
       addresses, the access serialises into k transactions -- a k-way conflict.

   THE WORKLOAD -- matrix transpose through shared memory
       This is NVIDIA's canonical example, from the CUDA Technical Blog post
       "An Efficient Matrix Transpose in CUDA C/C++" (Mark Harris); the kernel
       there is called `transposeCoalesced`.  Global memory is already handled
       correctly here: both the read and the write are fully coalesced, which
       is the whole point of staging through a tile.  The remaining problem is
       entirely inside shared memory.

   WHERE THE CONFLICT IS
           __shared__ float tile[TILE][TILE];          // TILE = 32
           ...
           odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

       For a fixed j, the 32 lanes of a warp vary threadIdx.x, so they read
       tile[0][c], tile[1][c], ... tile[31][c] -- one COLUMN of the tile.
       Consecutive rows are TILE = 32 floats apart, and 32 % 32 == 0, so every
       one of those addresses lands in the SAME bank:

           lane  0 -> tile[ 0][c] -> bank (0*32 + c) % 32 = c
           lane  1 -> tile[ 1][c] -> bank (1*32 + c) % 32 = c
           lane 31 -> tile[31][c] -> bank (31*32 + c) % 32 = c

       All 32 lanes, one bank, 32 different addresses: a full **32-way bank
       conflict**, so that single instruction is replayed 32 times.

       Note the WRITE into the tile, tile[threadIdx.y+j][threadIdx.x], is
       conflict-free -- lanes vary threadIdx.x, which is the fast dimension.
       Only the column read conflicts.

   FIX: see bank_conflict/transpose_opt.cu -- one character.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o transpose_naive transpose_naive.cu
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

__global__ void transpose_conflicts(float *odata, const float * __restrict__ idata,
                                    int width) {
    __shared__ float tile[TILE][TILE];          /* <-- stride 32 == 32 banks */

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(size_t)(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;        /* transposed block offset */
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        odata[(size_t)(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
        /*                                   ^^^ column read: 32-way conflict */
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

    printf("Shared-memory transpose with BANK CONFLICTS  [unoptimized]\n");
    printf("  %dx%d floats (%.1f MB), TILE=%d, blockDim %dx%d\n",
           width, width, (double)nb / 1e6, TILE, TILE, BLOCK_ROWS);
    printf("  shared tile: float[%d][%d] -> row stride %d floats -> %d banks apart\n",
           TILE, TILE, TILE, TILE % 32);

    /* ---- correctness ---- */
    transpose_conflicts<<<grd, blk>>>(dout, din, width);
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
        TIME_KERNEL(ms, 50, (transpose_conflicts<<<g2, blk>>>(ao, ai, w)));
        printf("  %4dx%-4d : %.4f ms   %6.1f GB/s\n",
               w, w, ms, 2.0 * (double)wb / (ms * 1e-3) / 1e9);
        cudaFree(ai); cudaFree(ao);
    }

    float ms = 0.0f;
    TIME_KERNEL(ms, 50, (transpose_conflicts<<<grd, blk>>>(dout, din, width)));
    printf("\nTiming (%dx%d): %.4f ms   %.1f GB/s\n",
           width, width, ms, 2.0 * (double)nb / (ms * 1e-3) / 1e9);
    printf("\nCompare with bank_conflict/transpose_opt.cu (tile padded to [%d][%d]).\n",
           TILE, TILE + 1);

    cudaFree(din); cudaFree(dout); free(h); free(g);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
Shared-memory transpose with BANK CONFLICTS  [unoptimized]
  2048x2048 floats (16.8 MB), TILE=32, blockDim 32x8
  shared tile: float[32][32] -> row stride 32 floats -> 0 banks apart

Correctness: PASS  (max error 0.000e+00)

Matrix-size sweep:
   512x512  : 0.0058 ms    359.4 GB/s
  1024x1024 : 0.0196 ms    429.0 GB/s
  2048x2048 : 0.0661 ms    507.7 GB/s

Nsight Compute, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum:
  512x512   :   253,952 conflicts
  2048x2048 : 4,063,232 conflicts
*/

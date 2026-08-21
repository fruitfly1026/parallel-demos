/* ============================================================================
   cuThermo pattern: MEMORY MISALIGNMENT -- the OPTIMIZED version.
   [paper Fig. 5(c) and Fig. 6]
   ----------------------------------------------------------------------------
   ############################################################################
   #  WHAT CHANGED vs misalignment/rowscale_naive.cu                      #
   ############################################################################

   ONE EXPRESSION, on the host side.  `row_scale` is byte-for-byte the kernel
   from the naive file -- it already takes the row stride as a parameter.  What
   changes is the stride it is given:

       naive: const int pitch = nx;                    // 1001 floats, 4004 B
       opt  : const int pitch = ((nx + 7) / 8) * 8;    // 1008 floats, 4032 B

   1008 is a multiple of 8 floats = 32 bytes, so every row now begins exactly
   on a sector boundary.  In the naive version row r started at r*1001 floats
   and drifted one float further out of alignment with each row, so nearly
   every row straddled an extra sector.

   This is exactly what cudaMallocPitch() does for you, and why it exists.  If
   you allocate 2-D data with plain cudaMalloc and a natural row length, you
   are choosing the naive version by default.

   THE COST: 0.7% more memory (1008 vs 1001 floats per row).  The padding
   columns are allocated and never read -- the kernel still loops c < nx.

   PART 1 of the output shows the same idea in NVIDIA's `offset` kernel: the
   naive file sweeps every offset 0..32 to expose the effect, and this file
   runs only the aligned offsets to confirm they are all equally fast.


   ############################################################################
   #  READ THIS BEFORE TIMING IT: the clock will NOT move.                    #
   ############################################################################

   On an RTX A4500 this pattern is clearly visible in the SECTOR COUNTS and
   completely invisible in wall-clock time.  Measured with Nsight Compute on
   the row_scale kernel:

                                  naive (pitch 1001)   opt (pitch 1008)
       sectors per LD request           4.28                3.43
       total global LD sectors      1,120,945             899,608   (-24.6%)
       DRAM bytes read                 32.81 MB            33.04 MB (same)
       kernel time                     0.1165 ms           0.1171 ms

   24.6% fewer sector transactions, and yet not one microsecond saved.  The
   reason is that misalignment does not add DRAM traffic -- the array is the
   same size either way -- it only adds L1<->L2 sector transactions.  This
   kernel is DRAM-bandwidth-bound at ~563 GB/s, so the extra sectors are
   absorbed and cost nothing.

   That is a genuine result, not a broken demo, and it is worth teaching:
     * a heat map showing a real pattern does not by itself mean a real
       speedup is available -- check what the kernel is bound by first;
     * the paper's own numbers agree.  SpMV is its misalignment case
       (Table 2, rowOffsets = Misaligned) and it has the SMALLEST improvement
       in the whole of Table 4: 1.85% on A4500, 1.63% on RTX 4090.
     * the cost would show on a kernel bound by L1 throughput rather than
       DRAM bandwidth -- compare bank_conflict/, where an L1-level effect is
       worth 1.87x at a cache-resident size and only 1.07x at a DRAM-bound one.

   WHAT THE HEAT MAP SHOWS
       naive  Sector 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |  total = 2
              Sector 4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |  total = 2
       opt    every sector | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |  total = 1

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o rowscale_opt rowscale_opt.cu
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

/* NVIDIA's `offset` kernel, unchanged -- run only at aligned offsets here. */
__global__ void offset_copy(float *out, const float * __restrict__ in, int s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + s;
    if (i < n) out[i] = in[i] + 1.0f;
}

/* Identical to the kernel in rowscale_naive.cu -- it always took the row
   stride as a parameter; only the value it is handed changes. */
__global__ void row_scale(float *y, const float * __restrict__ x,
                          int nx, int pitch) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (c < nx) {
        size_t idx = (size_t)r * pitch + c;
        y[idx] = x[idx] * 2.0f + 1.0f;
    }
}

int main(void) {
    printf("Memory misalignment removed  [optimized]\n");

    /* ================= part 1: aligned offsets only ================= */
    {
        const int n = 1 << 24;
        const size_t bytes = (size_t)(n + 33) * sizeof(float);
        float *h = (float *)malloc(bytes);
        float *g = (float *)malloc(bytes);
        for (int i = 0; i < n + 33; ++i) h[i] = (float)(i & 255);

        float *din, *dout;
        CUDA_CHECK(cudaMalloc(&din, bytes));
        CUDA_CHECK(cudaMalloc(&dout, bytes));
        CUDA_CHECK(cudaMemcpy(din, h, bytes, cudaMemcpyHostToDevice));
        const int blocks = (n + BLK - 1) / BLK;

        printf("\nCorrectness (offset kernel, aligned):\n");
        CUDA_CHECK(cudaMemset(dout, 0, bytes));
        offset_copy<<<blocks, BLK>>>(dout, din, 0, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(g, dout, bytes, cudaMemcpyDeviceToHost));
        double max_err = 0.0;
        for (int i = 0; i < n; ++i) {
            double e = fabs((double)g[i] - ((double)h[i] + 1.0));
            if (e > max_err) max_err = e;
        }
        printf("  offset=0  %s  (max error %.3e)\n",
               max_err < 1e-5 ? "PASS" : "FAIL", max_err);

        printf("\nAligned offsets only (every one a multiple of 8 floats):\n");
        float base = 1.0f;
        for (int s = 0; s <= 32; s += 8) {
            float ms = 0.0f;
            TIME_KERNEL(ms, 30, (offset_copy<<<blocks, BLK>>>(dout, din, s, n)));
            if (s == 0) base = ms;
            double gbps = 2.0 * (double)n * sizeof(float) / (ms * 1e-3) / 1e9;
            printf("  offset %2d floats (%3d B) : %.4f ms  %6.1f GB/s  %5.2fx\n",
                   s, s * 4, ms, gbps, ms / base);
        }
        cudaFree(din); cudaFree(dout); free(h); free(g);
    }

    /* ================= part 2: padded 2-D row pitch ================= */
    {
        const int rows  = 8192;
        const int nx    = 1001;                    /* same useful row length */
        const int pitch = ((nx + 7) / 8) * 8;      /* <-- padded to 1008 */
        size_t nb = (size_t)rows * pitch * sizeof(float);

        printf("\n2-D row pitch: rows=%d, nx=%d, pitch=%d floats (%d B, %s)\n",
               rows, nx, pitch, pitch * 4,
               (pitch % 8) ? "misaligned" : "every row on a sector boundary");
        printf("  array: %.1f MB  (%.1f%% more than unpadded, all of it never read)\n",
               (double)nb / 1e6, 100.0 * (pitch - nx) / nx);

        float *hx = (float *)malloc(nb), *hy = (float *)malloc(nb);
        for (size_t t = 0; t < (size_t)rows * pitch; ++t) hx[t] = (float)(t & 63);
        float *dx, *dy;
        CUDA_CHECK(cudaMalloc(&dx, nb)); CUDA_CHECK(cudaMalloc(&dy, nb));
        CUDA_CHECK(cudaMemcpy(dx, hx, nb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dy, 0, nb));

        dim3 blk(BLK, 1), grd((nx + BLK - 1) / BLK, rows);
        row_scale<<<grd, blk>>>(dy, dx, nx, pitch);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hy, dy, nb, cudaMemcpyDeviceToHost));
        double max_err = 0.0;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < nx; ++c) {
                size_t idx = (size_t)r * pitch + c;
                double e = fabs((double)hy[idx] - ((double)hx[idx] * 2.0 + 1.0));
                if (e > max_err) max_err = e;
            }
        float ms = 0.0f;
        TIME_KERNEL(ms, 50, (row_scale<<<grd, blk>>>(dy, dx, nx, pitch)));
        printf("  correctness: %s  (max error %.3e)\n",
               max_err < 1e-5 ? "PASS" : "FAIL", max_err);
        printf("  time: %.4f ms   %.1f GB/s\n",
               ms, 2.0 * (double)rows * nx * sizeof(float) / (ms * 1e-3) / 1e9);
        cudaFree(dx); cudaFree(dy); free(hx); free(hy);
    }

    printf("\nCompare with misalignment/rowscale_naive.cu (pitch = nx = 1001).\n");
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
Memory misalignment removed  [optimized]

Correctness (offset kernel, aligned):
  offset=0  PASS  (max error 0.000e+00)

Aligned offsets only:
  offset  0 floats (  0 B) : 0.2366 ms   567.3 GB/s   1.00x
  offset  8 floats ( 32 B) : 0.2346 ms   572.0 GB/s   0.99x
  offset 16 floats ( 64 B) : 0.2347 ms   572.0 GB/s   0.99x
  offset 24 floats ( 96 B) : 0.2347 ms   571.9 GB/s   0.99x
  offset 32 floats (128 B) : 0.2344 ms   572.5 GB/s   0.99x

2-D row pitch: rows=8192, nx=1001, pitch=1008 floats (4032 B, every row aligned)
  array: 33.0 MB  (0.7% more than unpadded, all of it never read)
  correctness: PASS  (max error 0.000e+00)
  time: 0.1171 ms   560.4 GB/s

Sectors per load request 4.28 -> 3.43 and total load sectors 1,120,945 ->
899,608 (-24.6%), with identical DRAM traffic and identical run time.
*/

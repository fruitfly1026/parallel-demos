/* ============================================================================
   cuThermo pattern: MEMORY MISALIGNMENT -- the UNOPTIMIZED version.
   [paper Fig. 5(c) and Fig. 6; Table 2 "SpMV / rowOffsets / Misaligned"]
   ----------------------------------------------------------------------------
   WHAT THE HEAT MAP LOOKS LIKE
       A run of sectors whose interior rows are ordinary -- each word touched
       by one warp -- but whose two BOUNDARY sectors are touched by TWO warps:

           Sector 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |   total = 2   <-- shared
           Sector 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |   total = 1
           Sector 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |   total = 1
           Sector 3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |   total = 1
           Sector 4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |   total = 2   <-- shared

   WHY IT COSTS YOU
       A warp's 32 floats are 128 bytes = exactly 4 sectors IF the base address
       is 32-byte aligned.  Off by even one float and the request spills into a
       fifth sector: 5 transactions instead of 4, and the two boundary sectors
       get pulled into L1 twice, once per warp.

   THE WORKLOAD -- a row-major 2-D array with an odd row length
       Rows are 1001 floats long.  1001 is not a multiple of 8, so the start of
       row r sits at r*1001 floats and drifts one float further out of
       alignment with every row.  Almost every row in the array therefore
       begins mid-sector.  This is the shape the paper reports for SpMV's
       rowOffsets, and it is what you get any time an array's minor dimension
       is a size the problem handed you rather than one you chose.

   PART 1 of the output is NVIDIA's canonical `offset` microbenchmark, which
   isolates the effect: the same copy kernel run at every base offset from 0 to
   32 floats.  Offsets that are multiples of 8 floats (32 bytes) are aligned;
   the rest are not.  It comes from the CUDA Technical Blog post "How to Access
   Global Memory Efficiently in CUDA C/C++ Kernels" and the CUDA C++ Best
   Practices Guide, "Coalesced Access to Global Memory".


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

   FIX: see misalignment/misalignment_opt.cu -- pad the row pitch.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o misalignment_naive misalignment_naive.cu
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

/* NVIDIA's `offset` kernel: shift the base address by s floats. */
__global__ void offset_copy(float *out, const float * __restrict__ in, int s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + s;
    if (i < n) out[i] = in[i] + 1.0f;
}

/* Row-major sweep over a 2-D array of `rows` rows, `nx` useful floats each,
   stored with a row stride of `pitch` floats. */
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
    printf("Memory misalignment  [unoptimized]\n");

    /* ================= part 1: NVIDIA offset sweep ================= */
    {
        const int n = 1 << 24;                   /* 16.7 M floats = 67 MB */
        const int maxs = 33;
        const size_t bytes = (size_t)(n + maxs) * sizeof(float);
        float *h = (float *)malloc(bytes);
        float *g = (float *)malloc(bytes);
        for (int i = 0; i < n + maxs; ++i) h[i] = (float)(i & 255);

        float *din, *dout;
        CUDA_CHECK(cudaMalloc(&din, bytes));
        CUDA_CHECK(cudaMalloc(&dout, bytes));
        CUDA_CHECK(cudaMemcpy(din, h, bytes, cudaMemcpyHostToDevice));
        const int blocks = (n + BLK - 1) / BLK;

        printf("\nCorrectness (offset kernel):\n");
        for (int s = 0; s <= 1; ++s) {
            CUDA_CHECK(cudaMemset(dout, 0, bytes));
            offset_copy<<<blocks, BLK>>>(dout, din, s, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(g, dout, bytes, cudaMemcpyDeviceToHost));
            double max_err = 0.0;
            for (int i = s; i < n; ++i) {
                double e = fabs((double)g[i] - ((double)h[i] + 1.0));
                if (e > max_err) max_err = e;
            }
            printf("  offset=%d  %s  (max error %.3e)\n",
                   s, max_err < 1e-5 ? "PASS" : "FAIL", max_err);
        }

        printf("\nOffset sweep -- 8 floats = one 32-byte sector:\n");
        float base = 1.0f;
        for (int s = 0; s <= 32; ++s) {
            float ms = 0.0f;
            TIME_KERNEL(ms, 30, (offset_copy<<<blocks, BLK>>>(dout, din, s, n)));
            if (s == 0) base = ms;
            double gbps = 2.0 * (double)n * sizeof(float) / (ms * 1e-3) / 1e9;
            printf("  offset %2d floats (%3d B) : %.4f ms  %6.1f GB/s  %5.2fx  %s\n",
                   s, s * 4, ms, gbps, ms / base, (s % 8 == 0) ? "<- aligned" : "");
        }
        cudaFree(din); cudaFree(dout); free(h); free(g);
    }

    /* ================= part 2: unpadded 2-D row pitch ================= */
    {
        const int rows  = 8192;
        const int nx    = 1001;              /* NOT a multiple of 8 floats */
        const int pitch = nx;                /* unpadded: row starts drift */
        size_t nb = (size_t)rows * pitch * sizeof(float);

        printf("\n2-D row pitch: rows=%d, nx=%d, pitch=%d floats (%d B, %s)\n",
               rows, nx, pitch, pitch * 4,
               (pitch % 8) ? "row starts drift across sectors" : "aligned");
        printf("  array: %.1f MB\n", (double)nb / 1e6);

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

    printf("\nCompare with misalignment/misalignment_opt.cu (padded row pitch).\n");
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
Memory misalignment  [unoptimized]

Correctness (offset kernel):
  offset=0  PASS  (max error 0.000e+00)
  offset=1  PASS  (max error 0.000e+00)

Offset sweep -- 8 floats = one 32-byte sector:
  offset  0 floats (  0 B) : 0.2370 ms   566.3 GB/s   1.00x  <- aligned
  offset  1 floats (  4 B) : 0.2346 ms   572.1 GB/s   0.99x
  ... every offset 1..32 within 1% of this ...
  offset 32 floats (128 B) : 0.2343 ms   572.8 GB/s   0.99x  <- aligned

  The sweep is FLAT. On this architecture a misaligned streaming copy costs
  nothing measurable; see the header for why, and for the sector counts that
  do show the pattern.

2-D row pitch: rows=8192, nx=1001, pitch=1001 floats (4004 B, row starts drift)
  array: 32.8 MB
  correctness: PASS  (max error 0.000e+00)
  time: 0.1165 ms   563.1 GB/s
*/

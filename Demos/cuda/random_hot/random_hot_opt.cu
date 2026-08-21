/* ============================================================================
   cuThermo pattern: RANDOM HOT SPOT -- the OPTIMIZED version.
   [paper Fig. 5(f); Table 2 "SpMV / spmv_csr / x / Random hot"]
   ----------------------------------------------------------------------------
   ############################################################################
   #  WHAT CHANGED vs random_hot/random_hot_naive.cu                          #
   ############################################################################

   THE DATA CHANGED.  THE KERNEL DID NOT.  That is the whole lesson of a
   RANDOM hot spot: the irregularity lives in the input, so no kernel rewrite
   fixes it.  spmv_coo below is byte-for-byte the kernel from the naive file.

   The nonzero columns are generated inside a band of +/- BAND around the
   diagonal instead of uniformly over [0, n):

       naive: colidx[j] = rng() % n;                          // anywhere
       opt  : colidx[j] = clamp(r + rng()%(2*BAND+1) - BAND, 0, n-1);

   Everything else about the matrix is identical -- same m, same n, same 8
   nonzeros per row, same 33.5 M total -- so the comparison is fair.  In a real
   code this reordering is what a bandwidth-reducing permutation (reverse
   Cuthill-McKee, or a graph/hypergraph partitioner) buys you.

   UNCHANGED ON PURPOSE: the atomicAdd into y.  Atomic contention is a real
   cost, but it is a WRITE problem, not the random hot spot under study, so it
   is kept identical on both sides.  At this problem size it turns out to be
   nearly free anyway: measured with the atomic removed entirely, the random
   case runs 4.166 ms vs 4.178 ms with it -- 0.3%.  The gather is everything.

   MEASURED (RTX A4500, 4.19 M x 4.19 M, 33.5 M nonzeros)
       random columns (naive)   : 4.1674 ms    8.05 Gnnz/s
       banded columns (this)    : 0.8871 ms   37.82 Gnnz/s   -> 4.70x

       The gap only appears once x outgrows the cache.  x here is 16.8 MB
       against this GPU's 4 MB L2, so a random gather genuinely reaches DRAM.
       At m = n = 2^20 (x = 4.2 MB, L2-resident) the same comparison gives only
       1.6x, and at 2^16 it gives 1.17x -- the pattern is real but invisible
       until the working set escapes the cache.  Worth remembering before
       concluding from a small test that a gather "is fine".

   ############################################################################
   #  A MEASURED NEGATIVE RESULT -- spmv_coo_smem                             #
   ############################################################################

   Once the columns are banded, every column a block touches is known to lie in
   a bounded window, so the obvious next move is to stage that slice of x in
   shared memory and let all 8 warps share it on-chip.  spmv_coo_smem below
   does exactly that, and it is CORRECT -- but it is SLOWER:

       banded, plain kernel       : 0.8871 ms   37.82 Gnnz/s
       banded + smem window       : 1.0055 ms   33.37 Gnnz/s   -> 0.88x

   Why: a block owns 256 nonzeros spanning 32 rows, so its window is only ~224
   floats (~900 bytes).  L1 already holds that comfortably.  The staging pass
   adds a full read of the window, two __syncthreads(), and a bounds test per
   nonzero, and buys nothing the cache was not already giving away.

   This is kept in the file deliberately.  "Put it in shared memory" is not a
   fix, it is a hypothesis -- and the reason to have a profiler is to test it.
   Compare with hotspot/hotspot_tiled.cu, where tiling DOES pay off (1.26x)
   because there the reuse is 1024 threads deep rather than 8 warps wide.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o random_hot_opt random_hot_opt.cu
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

#define BLK  256                 /* 8 warps per block */
#define BAND 96                  /* half-bandwidth of the reordered matrix */
#define WIN  512                 /* SMEM window capacity, floats (2 KB) */

/* Identical to the kernel in random_hot_naive.cu -- used here to measure what
   the data reordering alone is worth. */
__global__ void spmv_coo(int nnz, const int * __restrict__ rowidx,
                         const int * __restrict__ colidx,
                         const float * __restrict__ val,
                         const float * __restrict__ x, float *y) {
    int j = blockIdx.x * BLK + threadIdx.x;
    if (j < nnz)
        atomicAdd(&y[rowidx[j]], val[j] * x[colidx[j]]);
}

/* Same math, but the block's slice of x is staged in shared memory first. */
__global__ void spmv_coo_smem(int nnz, int n, const int * __restrict__ rowidx,
                              const int * __restrict__ colidx,
                              const float * __restrict__ val,
                              const float * __restrict__ x, float *y) {
    __shared__ float sx[WIN];
    __shared__ int s_lo, s_hi;

    int base = blockIdx.x * BLK;
    if (threadIdx.x == 0) {
        int last = base + BLK - 1; if (last > nnz - 1) last = nnz - 1;
        int lo = rowidx[base] - BAND;      if (lo < 0) lo = 0;
        int hi = rowidx[last] + BAND + 1;  if (hi > n) hi = n;
        if (hi - lo > WIN) hi = lo + WIN;  /* anything past the window falls back */
        s_lo = lo; s_hi = hi;
    }
    __syncthreads();
    int lo = s_lo, hi = s_hi;

    for (int t = threadIdx.x; t < hi - lo; t += BLK) sx[t] = x[lo + t];
    __syncthreads();

    int j = base + threadIdx.x;
    if (j < nnz) {
        int c = colidx[j];
        float xv = (c >= lo && c < hi) ? sx[c - lo] : x[c];
        atomicAdd(&y[rowidx[j]], val[j] * xv);
    }
}

static unsigned rng_state = 123456789u;
static unsigned rng(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;  return rng_state;
}

int main(void) {
    const int m = 1 << 22;
    const int n = m;
    const int nnz_row = 8;
    const int nnz = m * nnz_row;
    const int blocks = (nnz + BLK - 1) / BLK;

    int   *rowidx = (int   *)malloc((size_t)nnz * sizeof(int));
    int   *colidx = (int   *)malloc((size_t)nnz * sizeof(int));
    float *val    = (float *)malloc((size_t)nnz * sizeof(float));
    float *x      = (float *)malloc((size_t)n   * sizeof(float));
    float *y      = (float *)malloc((size_t)m   * sizeof(float));
    double *ref   = (double*)malloc((size_t)m   * sizeof(double));

    for (int i = 0; i < n; ++i) x[i] = (float)(i % 11) * 0.125f;
    for (int r = 0; r < m; ++r)
        for (int k = 0; k < nnz_row; ++k) {
            int j = r * nnz_row + k;
            rowidx[j] = r;
            int c = r + (int)(rng() % (unsigned)(2 * BAND + 1)) - BAND;
            if (c < 0) c = 0;
            if (c >= n) c = n - 1;
            colidx[j] = c;                                /* <-- banded */
            val[j]    = (float)(rng() % 7) * 0.25f + 0.5f;
        }

    for (int r = 0; r < m; ++r) ref[r] = 0.0;
    for (int j = 0; j < nnz; ++j) ref[rowidx[j]] += (double)val[j] * x[colidx[j]];

    int *drow, *dcol; float *dval, *dx, *dy;
    CUDA_CHECK(cudaMalloc(&drow, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dcol, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dval, (size_t)nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dx,   (size_t)n   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy,   (size_t)m   * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(drow, rowidx, (size_t)nnz*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dcol, colidx, (size_t)nnz*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dval, val,    (size_t)nnz*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx,   x,      (size_t)n  *sizeof(float), cudaMemcpyHostToDevice));

    printf("COO SpMV, BANDED column distribution  [optimized]\n");
    printf("  m=n=%d, %d nnz/row, %d nonzeros, blockDim=%d, half-bandwidth %d\n",
           m, nnz_row, nnz, BLK, BAND);
    printf("  SMEM window: %d floats (%d B) per block\n", WIN, WIN * 4);

    /* ---- correctness for both kernels ---- */
    printf("\nCorrectness:\n");
    for (int v = 0; v < 2; ++v) {
        CUDA_CHECK(cudaMemset(dy, 0, (size_t)m * sizeof(float)));
        if (v == 0) spmv_coo     <<<blocks, BLK>>>(nnz, drow, dcol, dval, dx, dy);
        else        spmv_coo_smem<<<blocks, BLK>>>(nnz, n, drow, dcol, dval, dx, dy);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(y, dy, (size_t)m * sizeof(float), cudaMemcpyDeviceToHost));
        double max_rel = 0.0;
        for (int r = 0; r < m; ++r) {
            double d = fabs((double)y[r] - ref[r]);
            double rr = d / (fabs(ref[r]) + 1e-6);
            if (rr > max_rel) max_rel = rr;
        }
        printf("  %-24s %s  (max rel error %.3e)\n",
               v == 0 ? "reordered (the fix)" : "reordered + smem window",
               max_rel < 1e-4 ? "PASS" : "FAIL", max_rel);
    }

    /* ---- parameter sweep: how much bandwidth can the window absorb? ---- */
    printf("\nNonzero-count sweep (reordered data, plain kernel):\n");
    for (int frac = 4; frac >= 1; frac /= 2) {
        int sub = nnz / frac;
        int sb = (sub + BLK - 1) / BLK;
        float ms = 0.0f;
        CUDA_CHECK(cudaMemset(dy, 0, (size_t)m * sizeof(float)));
        TIME_KERNEL(ms, 30, (spmv_coo<<<sb, BLK>>>(sub, drow, dcol, dval, dx, dy)));
        printf("  %8d nnz : %.4f ms   %6.2f Gnnz/s\n",
               sub, ms, (double)sub / (ms * 1e-3) / 1e9);
    }

    float t_a = 0.0f, t_b = 0.0f;
    CUDA_CHECK(cudaMemset(dy, 0, (size_t)m * sizeof(float)));
    TIME_KERNEL(t_a, 50, (spmv_coo     <<<blocks, BLK>>>(nnz, drow, dcol, dval, dx, dy)));
    CUDA_CHECK(cudaMemset(dy, 0, (size_t)m * sizeof(float)));
    TIME_KERNEL(t_b, 50, (spmv_coo_smem<<<blocks, BLK>>>(nnz, n, drow, dcol, dval, dx, dy)));
    printf("\nTiming (full matrix):\n");
    printf("  reordered data, same kernel as naive : %.4f ms   %.2f Gnnz/s   <-- the fix\n",
           t_a, (double)nnz / (t_a * 1e-3) / 1e9);
    printf("  reordered data + smem window         : %.4f ms   %.2f Gnnz/s   (%.2fx -- see\n"
           "                                         the header: staging does not pay here)\n",
           t_b, (double)nnz / (t_b * 1e-3) / 1e9, t_a / t_b);
    printf("\nCompare with random_hot/random_hot_naive.cu (same kernel, random columns).\n");

    cudaFree(drow); cudaFree(dcol); cudaFree(dval); cudaFree(dx); cudaFree(dy);
    free(rowidx); free(colidx); free(val); free(x); free(y); free(ref);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
COO SpMV, BANDED column distribution  [optimized]
  m=n=4194304, 8 nnz/row, 33554432 nonzeros, blockDim=256, half-bandwidth 96
  SMEM window: 512 floats (2048 B) per block

Correctness:
  reordered (the fix)      PASS  (max rel error 0.000e+00)
  reordered + smem window  PASS  (max rel error 0.000e+00)

Nonzero-count sweep (reordered data, plain kernel):
   8388608 nnz : 0.2243 ms    37.40 Gnnz/s
  16777216 nnz : 0.4452 ms    37.68 Gnnz/s
  33554432 nnz : 0.8869 ms    37.83 Gnnz/s

Timing (full matrix):
  reordered data, same kernel as naive : 0.8869 ms   37.83 Gnnz/s   <-- the fix
  reordered data + smem window         : 1.0054 ms   33.37 Gnnz/s   (0.88x)

vs random_hot_naive.cu: 4.1674 -> 0.8869 ms = 4.70x, from changing the data
alone. The shared-memory window is a measured negative; see the header.
*/

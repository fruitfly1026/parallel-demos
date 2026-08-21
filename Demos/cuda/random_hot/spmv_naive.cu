/* ============================================================================
   cuThermo pattern: RANDOM HOT SPOT -- the UNOPTIMIZED version.
   [paper Fig. 5(f); Table 2 "SpMV / spmv_csr / x / Random hot"]
   ----------------------------------------------------------------------------
   WHAT THE HEAT MAP LOOKS LIKE
       Hot, but ragged.  Per-word distinct-warp counts inside a sector are all
       over the place while the sector totals stay high:

           Sector 0 | 2 | 1 | 5 | 3 | 2 | 2 | 1 | 2 |   total = 8
           Sector 1 | 4 | 4 | 2 | 5 | 7 | 3 | 1 | 3 |   total = 8
           Sector 2 | 3 | 2 | 0 | 4 | 3 | 3 | 3 | 8 |   total = 8

       Compare with hotspot/, where every entry is the SAME number.  Uniform
       hot means structured reuse you can tile.  RANDOM hot means the reuse is
       real but input-dependent, so the fix is a data-layout change, not a
       kernel rewrite.

   THE WORKLOAD -- SpMV in COO format
       One thread per nonzero:

           y[rowidx[j]] += val[j] * x[colidx[j]]

       rowidx, colidx and val are all read with consecutive j, so they are
       perfectly coalesced.  x[colidx[j]] is the ONLY irregular access in the
       kernel, which is what makes COO a cleaner demonstration than CSR: there
       is no per-thread inner loop, so every thread does exactly the same
       amount of work and no load imbalance can be confused for a memory
       effect.

   ABOUT THE atomicAdd
       Scattering into y needs an atomic, and atomic contention is a REAL cost
       -- but it is write contention, not the random hot spot we are studying.
       spmv_opt.cu keeps the identical atomicAdd so that cost is the same
       on both sides and the measured difference isolates the x gather.

   PROVENANCE
       A self-contained COO SpMV written for this demo, not lifted from a
       benchmark suite.  The pattern is the one the paper reports for SpMV's
       x vector; the timings are this benchmark's own.

   FIX: see random_hot/spmv_opt.cu.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o spmv_naive spmv_naive.cu
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

#define BLK 256                  /* 8 warps per block */

/* COO SpMV, one thread per nonzero.  x[colidx[j]] is the random gather. */
__global__ void spmv_coo(int nnz, const int * __restrict__ rowidx,
                         const int * __restrict__ colidx,
                         const float * __restrict__ val,
                         const float * __restrict__ x, float *y) {
    int j = blockIdx.x * BLK + threadIdx.x;
    if (j < nnz)
        atomicAdd(&y[rowidx[j]], val[j] * x[colidx[j]]);   /* <-- random gather on x */
}

static unsigned rng_state = 123456789u;
static unsigned rng(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;  return rng_state;
}

int main(void) {
    const int m = 1 << 22;                   /* 4.19 M rows and columns */
    const int n = m;
    const int nnz_row = 8;
    const int nnz = m * nnz_row;             /* 2.1 M nonzeros */
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
            colidx[j] = (int)(rng() % (unsigned)n);       /* <-- scattered */
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

    printf("COO SpMV, RANDOM column distribution  [unoptimized]\n");
    printf("  m=n=%d, %d nnz/row, %d nonzeros, blockDim=%d\n", m, nnz_row, nnz, BLK);
    printf("  x is %.1f MB -- past this GPU's 4 MB L2, so a random gather reaches DRAM\n",
           (double)n * sizeof(float) / 1e6);

    /* ---- correctness ---- */
    CUDA_CHECK(cudaMemset(dy, 0, (size_t)m * sizeof(float)));
    spmv_coo<<<blocks, BLK>>>(nnz, drow, dcol, dval, dx, dy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(y, dy, (size_t)m * sizeof(float), cudaMemcpyDeviceToHost));
    double max_rel = 0.0;
    for (int r = 0; r < m; ++r) {
        double d = fabs((double)y[r] - ref[r]);
        double rr = d / (fabs(ref[r]) + 1e-6);
        if (rr > max_rel) max_rel = rr;
    }
    printf("\nCorrectness: %s  (max rel error %.3e)\n",
           max_rel < 1e-4 ? "PASS" : "FAIL", max_rel);

    /* ---- parameter sweep: blockDim is fixed by BLK, so sweep nnz instead ---- */
    printf("\nNonzero-count sweep (same random layout):\n");
    for (int frac = 4; frac >= 1; frac /= 2) {
        int sub = nnz / frac;
        int sb = (sub + BLK - 1) / BLK;
        float ms = 0.0f;
        CUDA_CHECK(cudaMemset(dy, 0, (size_t)m * sizeof(float)));
        TIME_KERNEL(ms, 30, (spmv_coo<<<sb, BLK>>>(sub, drow, dcol, dval, dx, dy)));
        printf("  %8d nnz : %.4f ms   %6.2f Gnnz/s\n",
               sub, ms, (double)sub / (ms * 1e-3) / 1e9);
    }

    float ms = 0.0f;
    CUDA_CHECK(cudaMemset(dy, 0, (size_t)m * sizeof(float)));
    TIME_KERNEL(ms, 50, (spmv_coo<<<blocks, BLK>>>(nnz, drow, dcol, dval, dx, dy)));
    printf("\nTiming (full matrix): %.4f ms   %.2f Gnnz/s\n",
           ms, (double)nnz / (ms * 1e-3) / 1e9);
    printf("\nCompare with random_hot/spmv_opt.cu (same kernel, reordered data).\n");

    cudaFree(drow); cudaFree(dcol); cudaFree(dval); cudaFree(dx); cudaFree(dy);
    free(rowidx); free(colidx); free(val); free(x); free(y); free(ref);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
COO SpMV, RANDOM column distribution  [unoptimized]
  m=n=4194304, 8 nnz/row, 33554432 nonzeros, blockDim=256
  x is 16.8 MB -- past this GPU's 4 MB L2, so a random gather reaches DRAM

Correctness: PASS  (max rel error 0.000e+00)

Nonzero-count sweep (same random layout):
   8388608 nnz : 1.0484 ms     8.00 Gnnz/s
  16777216 nnz : 2.0913 ms     8.02 Gnnz/s
  33554432 nnz : 4.1751 ms     8.04 Gnnz/s

Timing (full matrix): 4.1674 ms   8.05 Gnnz/s

Throughput is flat in nnz -- every gather is an independent DRAM round trip,
so there is no economy of scale to be had. vs spmv_opt.cu: 4.1674 /
0.8869 = 4.70x slower.
*/

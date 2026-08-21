/* ============================================================================
   cuThermo pattern: HOT SPOT  -- the UNOPTIMIZED version.
   [paper Fig. 5(e); Table 2 "GEMM / gemm_v01 / B / Hot"; Table 4 "26.07%"]
   ----------------------------------------------------------------------------
   SOURCE
       Kernel `gemm_v01` is taken verbatim from
       https://github.com/leimao/CUDA-GEMM-Optimization
           src/01_coalesced_global_memory_access.cu
       reference [20] of the cuThermo paper.  This is the SAME kernel as
       false_sharing/gemm_opt.cu -- it is the fixed version of the false-sharing
       problem, and the starting point for the next one.

   THE INEFFICIENCY THAT IS LEFT
       Coalescing is now perfect, but there is no data REUSE: every thread
       re-reads A and B straight from global memory on every k step.

       B[k_idx*ldb + C_col_idx] depends only on threadIdx.x, so it is byte-for-
       byte identical for all 32 warps of the block.  All 32 warps hammer the
       same 4 sectors of B at every k:

           Sector 0 | 32 | 32 | 32 | 32 | 32 | 32 | 32 | 32 |   total = 32
           Sector 1 | 32 | 32 | 32 | 32 | 32 | 32 | 32 | 32 |   total = 32

       Uniformly hot in the words AND in the sector -- the Fig. 5(e) signature,
       and precisely what Table 2 records as "gemm_v01 / B / Hot".  The reuse
       is real, it is just being served by L1 instead of by shared memory.

   MEASURED (RTX A4500, N=1024, Nsight Compute)
       sectors per global LOAD request : 2.50   (coalescing is fine)
       total global load sectors       : 2.63 M (the traffic is not)

   FIX: see hotspot/gemm_tiled.cu.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o gemm_coalesced gemm_coalesced.cu
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

/* ---- kernel, unchanged from leimao src/01_coalesced_global_memory_access.cu ---- */
template <typename T>
__global__ void gemm_v01(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            /* every k step goes back to global memory for both operands */
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

#define LAUNCH(A, B, Cm, N)                                                   \
    gemm_v01<float><<<dim3(((unsigned)(N)+31u)/32u, ((unsigned)(N)+31u)/32u),  \
                      dim3(32u, 32u)>>>((size_t)(N), (size_t)(N), (size_t)(N),\
                      1.0f, (A), (size_t)(N), (B), (size_t)(N), 0.0f,         \
                      (Cm), (size_t)(N))

#define KERNEL_TITLE "GEMM v01 -- coalesced but B is a HOT SPOT  [unoptimized]"
#define COMPARE_HINT "Compare with hotspot/gemm_tiled.cu (gemm_v02: 32x32x32 shared-memory block tiling)."

/* ------------------------------------------------------------------ harness */
static void gemm_cpu(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int t = 0; t < N; t++) s += (double)A[(size_t)i*N+t] * B[(size_t)t*N+j];
            C[(size_t)i*N+j] = (float)s;
        }
}

static float time_it(const float *dA, const float *dB, float *dC, int N, int reps) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    LAUNCH(dA, dB, dC, N);                                  /* warm-up */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int r = 0; r < reps; r++) LAUNCH(dA, dB, dC, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / (float)reps;
}

int main(void) {
    printf("%s\n", KERNEL_TITLE);
    printf("blockDim 32x32; alpha=1, beta=0 so C = A*B\n");

    /* ---- (a) CORRECTNESS at N=512 against a CPU reference ---- */
    {
        int N = 512;
        size_t nb = (size_t)N*N*sizeof(float);
        float *hA = (float*)malloc(nb), *hB = (float*)malloc(nb);
        float *hC = (float*)malloc(nb), *hR = (float*)malloc(nb);
        srand(1);
        for (int i = 0; i < N*N; i++) { hA[i] = (float)((rand()%5)-2); hB[i] = (float)((rand()%5)-2); }
        gemm_cpu(hA, hB, hR, N);                     /* integer-valued => exact */

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, nb)); CUDA_CHECK(cudaMalloc(&dB, nb)); CUDA_CHECK(cudaMalloc(&dC, nb));
        CUDA_CHECK(cudaMemcpy(dA, hA, nb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, nb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dC, 0, nb));
        LAUNCH(dA, dB, dC, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hC, dC, nb, cudaMemcpyDeviceToHost));

        double max_err = 0.0; int bad = 0, first = -1;
        for (int i = 0; i < N*N; i++) {
            double d = fabs((double)hC[i] - (double)hR[i]);
            if (d > max_err) max_err = d;
            if (d > 1e-3) { if (first < 0) first = i; bad++; }
        }
        printf("\nCorrectness (N=%d, integer-valued so exact):\n", N);
        printf("  %s  (max error %.3e, %d/%d elements wrong)\n",
               bad == 0 ? "PASS" : "FAIL", max_err, bad, N*N);
        if (bad) printf("  first mismatch at %d: got %f, expected %f\n", first, hC[first], hR[first]);
        free(hA); free(hB); free(hC); free(hR);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    /* ---- (b) PARAMETER VARYING: N = 256 / 512 / 1024 ---- */
    printf("\nSize sweep:\n");
    for (int N : {256, 512, 1024}) {
        size_t nb = (size_t)N*N*sizeof(float);
        float *hA = (float*)malloc(nb), *hC = (float*)malloc(nb);
        for (int i = 0; i < N*N; i++) hA[i] = 1.0f;          /* all ones => C[i] == N */
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, nb)); CUDA_CHECK(cudaMalloc(&dB, nb)); CUDA_CHECK(cudaMalloc(&dC, nb));
        CUDA_CHECK(cudaMemcpy(dA, hA, nb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hA, nb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dC, 0, nb));
        float ms = time_it(dA, dB, dC, N, 10);
        CUDA_CHECK(cudaMemcpy(hC, dC, nb, cudaMemcpyDeviceToHost));
        bool ok = fabs((double)hC[0] - (double)N) < 1e-3;
        double gflop = 2.0*(double)N*N*N / 1e9;
        printf("  N=%4d : %9.3f ms  %7.1f GFLOP/s  (%s)\n",
               N, ms, gflop/(ms*1e-3), ok ? "ok" : "BAD");
        free(hA); free(hC);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    printf("\n%s\n", COMPARE_HINT);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
GEMM v01 -- coalesced but B is a HOT SPOT  [unoptimized]
blockDim 32x32; alpha=1, beta=0 so C = A*B

Correctness (N=512, integer-valued so exact):
  PASS  (max error 0.000e+00, 0/262144 elements wrong)

Size sweep:
  N= 256 :     0.043 ms    771.5 GFLOP/s  (ok)
  N= 512 :     0.204 ms   1314.7 GFLOP/s  (ok)
  N=1024 :     1.551 ms   1384.8 GFLOP/s  (ok)

vs gemm_tiled.cu at N=1024: 1.551 / 1.235 = 1.26x slower.
*/

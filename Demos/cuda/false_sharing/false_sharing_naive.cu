/* ============================================================================
   cuThermo pattern: MEMORY FALSE SHARING  -- the UNOPTIMIZED version.
   [paper Fig. 5(b); Sec. 6.1; Table 2 "GEMM / gemm_v00"; Table 4 "721.79%"]
   ----------------------------------------------------------------------------
   SOURCE
       Kernel `gemm_v00` is taken verbatim from
       https://github.com/leimao/CUDA-GEMM-Optimization
           src/00_non_coalesced_global_memory_access.cu
       which is reference [20] of the cuThermo paper and the origin of the
       paper's Listing 2.  Only the host harness below is new.

   THE INEFFICIENCY
       threadIdx.x feeds the C ROW index.  With blockDim 32x32 a warp is one
       row of threadIdx.x at fixed threadIdx.y, so the 32 lanes of a warp are
       32 DIFFERENT rows of C:

           A[C_row_idx*lda + k_idx]   lanes stride lda apart  -> 32 sectors
           C[C_row_idx*ldc + C_col_idx]  same                 -> 32 sectors
           B[k_idx*ldb + C_col_idx]   constant across a warp  -> broadcast

       B and C are where the FALSE SHARING shows up, and it is an ACROSS-WARP
       effect.  Warp w of the block has threadIdx.y = w, hence C_col_idx =
       blockIdx.y*32 + w.  So the 32 warps together touch 32 consecutive words
       = 4 sectors, and each 32-byte sector has its 8 words claimed by 8
       DIFFERENT warps:

           Sector | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |   sector total = 8
                    ^ each word one warp        ^ but the sector is 8x hotter

       Coalescing merges requests only WITHIN a warp, so those 8 words cost 8
       separate sector transactions instead of 1.  That is exactly the paper's
       Fig. 5(b), and Table 2 lists B and C of gemm_v00 as "False sharing".

   MEASURED (RTX A4500, N=1024, Nsight Compute)
       sectors per global LOAD request  : 16.53
       sectors per global STORE request : 32
       total global load sectors        : 17.4 M

   FIX: see false_sharing/false_sharing_opt.cu -- it is a two-line change.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o false_sharing_naive false_sharing_naive.cu
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

/* ---- kernel, unchanged from leimao src/00_non_coalesced_global_memory_access.cu ---- */
template <typename T>
__global__ void gemm_v00(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

/* leimao's launch config for v00: grid.x covers m, grid.y covers n. */
#define LAUNCH(A, B, Cm, N)                                                   \
    gemm_v00<float><<<dim3(((unsigned)(N)+31u)/32u, ((unsigned)(N)+31u)/32u),  \
                      dim3(32u, 32u)>>>((size_t)(N), (size_t)(N), (size_t)(N),\
                      1.0f, (A), (size_t)(N), (B), (size_t)(N), 0.0f,         \
                      (Cm), (size_t)(N))

#define KERNEL_TITLE "GEMM v00 -- non-coalesced / FALSE SHARING on B and C  [unoptimized]"
#define COMPARE_HINT "Compare with false_sharing/false_sharing_opt.cu (gemm_v01: two index lines swapped)."

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
GEMM v00 -- non-coalesced / FALSE SHARING on B and C  [unoptimized]
blockDim 32x32; alpha=1, beta=0 so C = A*B

Correctness (N=512, integer-valued so exact):
  PASS  (max error 0.000e+00, 0/262144 elements wrong)

Size sweep:
  N= 256 :     0.338 ms     99.4 GFLOP/s  (ok)
  N= 512 :     1.668 ms    160.9 GFLOP/s  (ok)
  N=1024 :    12.613 ms    170.3 GFLOP/s  (ok)

vs false_sharing_opt.cu at N=1024: 12.613 / 1.550 = 8.14x slower.
*/

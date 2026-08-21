/* ============================================================================
   cuThermo pattern: HOT SPOT  -- the OPTIMIZED version.
   [paper Fig. 5(e); Table 4: gemm_v01 -> 26.07% on A4500, 23.28% on RTX 4090]
   ----------------------------------------------------------------------------
   SOURCE
       Kernel `gemm_v02` follows
       https://github.com/leimao/CUDA-GEMM-Optimization
           src/02_2d_block_tiling.cu
       reference [20] of the cuThermo paper.  leimao's version calls a shared
       helper, load_data_from_global_memory_to_shared_memory(), out of
       include/cuda_gemm_utils.cuh; that helper is inlined here so this file
       builds standalone with plain nvcc.  With BLOCK_TILE 32x32x32 and 1024
       threads the helper's outer load loop runs exactly one iteration, so the
       inlined form below is equivalent.

   ############################################################################
   #  WHAT CHANGED vs hotspot/hotspot_coalesced.cu                            #
   ############################################################################

   The index mapping is UNCHANGED -- C_col_idx still comes from threadIdx.x,
   C_row_idx from threadIdx.y, so everything false_sharing/ fixed stays fixed.
   What is added is 2-D block tiling in shared memory:

   1. Two shared-memory tiles are declared:
          __shared__ float At[BT][BT];        // BT = 32
          __shared__ float Bt[BT][BT];

   2. The single k loop

          for (size_t k_idx = 0; k_idx < k; ++k_idx)
              sum += A[C_row_idx*lda + k_idx] * B[k_idx*ldb + C_col_idx];

      becomes a two-level loop: an outer loop over ceil(k/BT) tiles, and an
      inner loop of BT steps that reads only from shared memory:

          for (ti = 0; ti < ntiles; ++ti) {
              <cooperatively load one 32x32 tile of A and of B>   // global -> smem
              __syncthreads();
              for (ki = 0; ki < BT; ++ki)
                  sum += At[threadIdx.y][ki] * Bt[ki][threadIdx.x];   // smem only
              __syncthreads();
          }

   3. The tile loads are addressed by a THREAD-LINEAR index
          tl = threadIdx.y * blockDim.x + threadIdx.x
      rather than by (threadIdx.y, threadIdx.x) directly.  That decouples the
      tile shape from the block shape, which is what lets leimao's later
      versions (v03-v07) use thread tiling.

   4. The tile loads are BOUNDARY CHECKED and zero-filled:
          At[ar][ac] = (A_r < m && A_c < k) ? A[A_r*lda + A_c] : 0.0f;
      so the kernel is correct for any m, n, k -- not only multiples of 32.

   5. Two __syncthreads() per tile: one after the load, one after the compute
      (the second stops the next iteration's load from overwriting a tile that
      other warps are still reading).

   WHY THAT FIXES THE HOT SPOT
       In hotspot_coalesced.cu every one of the block's 32 warps re-read the
       same sectors of B out of global memory at every k step (word temperature
       32, sector temperature 32).  Now each tile is fetched from global memory
       ONCE per block, by one cooperative coalesced load, and the 32 warps
       share it on-chip.  The hot region moves off the global-memory heat map
       entirely; what is left in global memory is a cool, strictly streaming
       access.

   MEASURED (RTX A4500, N=1024, Nsight Compute)
                                    hotspot_coalesced.cu   hotspot_tiled.cu
       sectors per global LD request        2.50                4.00
       total global load sectors            2,629,632           139,264   (18.9x fewer)
       kernel time                          1.552 ms            1.235 ms
                                                                -> 1.26x = +25.7%
       (the paper reports 26.07% for this same step, Table 4)

       Note the per-request ratio RISES to 4.00 -- that is not a regression.
       4 sectors is a full 128-byte warp-wide load, i.e. perfect. The number
       that matters is the total, which falls by 18.9x.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o hotspot_tiled hotspot_tiled.cu
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

#define BT 32          /* BLOCK_TILE_SIZE_X == _Y == _K, as in leimao's v02 */

template <typename T>
__global__ void gemm_v02(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    /* Cache a tile of A and B in shared memory for data reuse. */
    __shared__ T A_thread_block_tile[BT][BT];
    __shared__ T B_thread_block_tile[BT][BT];

    size_t const num_thread_block_tiles{(k + BT - 1) / BT};

    T sum{static_cast<T>(0)};
    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        /* --- inlined load_data_from_global_memory_to_shared_memory --- */
        size_t const tile_row{thread_linear_idx / BT};
        size_t const tile_col{thread_linear_idx % BT};

        size_t const A_row_idx{blockIdx.y * BT + tile_row};
        size_t const A_col_idx{thread_block_tile_idx * BT + tile_col};
        A_thread_block_tile[tile_row][tile_col] =
            (A_row_idx < m && A_col_idx < k) ? A[A_row_idx * lda + A_col_idx]
                                             : static_cast<T>(0);

        size_t const B_row_idx{thread_block_tile_idx * BT + tile_row};
        size_t const B_col_idx{blockIdx.x * BT + tile_col};
        B_thread_block_tile[tile_row][tile_col] =
            (B_row_idx < k && B_col_idx < n) ? B[B_row_idx * ldb + B_col_idx]
                                             : static_cast<T>(0);
        /* ------------------------------------------------------------- */
        __syncthreads();

#pragma unroll
        for (size_t k_i{0U}; k_i < BT; ++k_i)
        {
            sum += A_thread_block_tile[threadIdx.y][k_i] *
                   B_thread_block_tile[k_i][threadIdx.x];
        }
        __syncthreads();
    }

    if (C_row_idx < m && C_col_idx < n)
    {
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

#define LAUNCH(A, B, Cm, N)                                                   \
    gemm_v02<float><<<dim3(((unsigned)(N)+31u)/32u, ((unsigned)(N)+31u)/32u),  \
                      dim3(32u, 32u)>>>((size_t)(N), (size_t)(N), (size_t)(N),\
                      1.0f, (A), (size_t)(N), (B), (size_t)(N), 0.0f,         \
                      (Cm), (size_t)(N))

#define KERNEL_TITLE "GEMM v02 -- 32x32x32 shared-memory block tiling  [optimized]"
#define COMPARE_HINT "Compare with hotspot/hotspot_coalesced.cu (gemm_v01, B is hot)."

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
GEMM v02 -- 32x32x32 shared-memory block tiling  [optimized]
blockDim 32x32; alpha=1, beta=0 so C = A*B

Correctness (N=512, integer-valued so exact):
  PASS  (max error 0.000e+00, 0/262144 elements wrong)

Size sweep:
  N= 256 :     0.035 ms    963.8 GFLOP/s  (ok)
  N= 512 :     0.163 ms   1648.8 GFLOP/s  (ok)
  N=1024 :     1.235 ms   1739.5 GFLOP/s  (ok)

vs hotspot_coalesced.cu at N=1024: 1.551 -> 1.235 ms = 1.26x = +25.7%
(paper Table 4 reports 26.07% for this step on the A4500).

Correct at sizes that are not a multiple of the tile, unlike the TILE=16
kernel in the old 04_matmul.cu:  N=1000 -> PASS here, FAIL there.
*/

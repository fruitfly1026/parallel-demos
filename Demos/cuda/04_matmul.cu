/* Square matrix multiply C = A*B, to the professor's spec:
     (a) CORRECTNESS CHECK  — both GPU kernels vs a CPU reference, PASS/FAIL.
     (b) PARAMETER VARYING  — the tiled kernel is run at N = 256 / 512 / 1024.
     (c) INEFFICIENT vs OPTIMIZED — naive global-memory matmul vs a shared-
                              memory tiled matmul, timed with cudaEvents.

   Two kernels:
     matmul_naive  — each thread reads every A/B element it needs straight from
                     global memory. Element A[row][k] is re-read by all TILE
                     threads of the row-block; B[k][col] by all TILE threads of
                     the col-block — O(N) redundant global loads per output.
                     INEFFICIENCY: no data reuse; the kernel is DRAM-bandwidth
                     bound with a ~1/TILE effective cache hit rate.
     matmul_tiled  — the block cooperatively stages a TILExTILE block of A and B
                     into shared memory once, then every thread reuses it TILE
                     times. FIX: shared-memory tiling cuts global loads by ~TILE.

   Build: nvcc -O3 -arch=sm_86 -o 04_matmul 04_matmul.cu                      */
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

#define TILE 16

__global__ void matmul_naive(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float s = 0.0f;
        for (int k = 0; k < N; k++) s += A[row*N + k] * B[k*N + col];
        C[row*N + col] = s;
    }
}

__global__ void matmul_tiled(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float s = 0.0f;
    for (int t = 0; t < N / TILE; t++) {                 /* assumes TILE | N */
        As[threadIdx.y][threadIdx.x] = A[row*N + t*TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t*TILE + threadIdx.y)*N + col];
        __syncthreads();
        for (int k = 0; k < TILE; k++) s += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row*N + col] = s;
}

/* CPU reference (used for correctness; kept at N<=512 so it stays quick). */
static void matmul_cpu(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++) s += (double)A[i*N+k] * B[k*N+j];
            C[i*N+j] = (float)s;
        }
}

static float time_matmul(void (*k)(const float*,const float*,float*,int),
                         const float *A, const float *B, float *C, int N) {
    dim3 threads(TILE, TILE), blocks(N/TILE, N/TILE);
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    k<<<blocks, threads>>>(A, B, C, N);          /* warm-up */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int r = 0; r < 10; r++) k<<<blocks, threads>>>(A, B, C, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / 10.0f;
}

static double max_abs_diff(const float *X, const float *Y, int N) {
    double m = 0.0;
    for (int i = 0; i < N*N; i++) { double d = fabs((double)X[i]-Y[i]); if (d>m) m=d; }
    return m;
}

int main(void) {
    /* ---- (a) CORRECTNESS at N=512: both kernels vs the CPU reference. ---- */
    int N = 512;                                 /* divisible by TILE (16) */
    size_t bytes = (size_t)N*N*sizeof(float);
    float *A, *B, *C;
    CUDA_CHECK(cudaMallocManaged(&A, bytes));
    CUDA_CHECK(cudaMallocManaged(&B, bytes));
    CUDA_CHECK(cudaMallocManaged(&C, bytes));
    srand(1);
    for (int i = 0; i < N*N; i++) { A[i] = (rand()%5)-2; B[i] = (rand()%5)-2; }
    float *ref = (float*)malloc(bytes);
    matmul_cpu(A, B, ref, N);

    matmul_naive<<<dim3(N/TILE,N/TILE), dim3(TILE,TILE)>>>(A, B, C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double err_naive = max_abs_diff(C, ref, N);

    matmul_tiled<<<dim3(N/TILE,N/TILE), dim3(TILE,TILE)>>>(A, B, C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double err_tiled = max_abs_diff(C, ref, N);

    printf("Correctness (N=%d, integer-valued so exact):\n", N);
    printf("  naive : %s  (max error %.3e)\n", err_naive < 1e-3 ? "PASS":"FAIL", err_naive);
    printf("  tiled : %s  (max error %.3e)\n", err_tiled < 1e-3 ? "PASS":"FAIL", err_tiled);
    free(ref);
    cudaFree(A); cudaFree(B); cudaFree(C);

    /* ---- (b) PARAMETER VARYING + (c) INEFFICIENT vs OPTIMIZED. ----
       For each size run both kernels (all-ones inputs so C[i]=N exactly, an
       easy self-check) and report the tiled speedup. */
    printf("\nSize sweep (naive vs tiled, TILE=%d):\n", TILE);
    for (int n : {256, 512, 1024}) {
        size_t nb = (size_t)n*n*sizeof(float);
        float *a, *b, *c;
        CUDA_CHECK(cudaMallocManaged(&a, nb));
        CUDA_CHECK(cudaMallocManaged(&b, nb));
        CUDA_CHECK(cudaMallocManaged(&c, nb));
        for (int i = 0; i < n*n; i++) { a[i] = 1.0f; b[i] = 1.0f; }

        float ms_naive = time_matmul(matmul_naive, a, b, c, n);
        bool ok_n = fabs(c[0] - (float)n) < 1e-3;
        float ms_tiled = time_matmul(matmul_tiled, a, b, c, n);
        bool ok_t = fabs(c[0] - (float)n) < 1e-3;
        double gflop = 2.0*(double)n*n*n / 1e9;
        printf("  N=%4d : naive %8.3f ms (%6.1f GFLOP/s, %s)  "
               "tiled %8.3f ms (%6.1f GFLOP/s, %s)  speedup %.2fx\n",
               n, ms_naive, gflop/(ms_naive*1e-3), ok_n?"ok":"BAD",
               ms_tiled, gflop/(ms_tiled*1e-3), ok_t?"ok":"BAD",
               ms_naive/ms_tiled);
        cudaFree(a); cudaFree(b); cudaFree(c);
    }
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), Xeon w7-2495X 48c:
Correctness (N=512, integer-valued so exact):
  naive : PASS  (max error 0.000e+00)
  tiled : PASS  (max error 0.000e+00)

Size sweep (naive vs tiled, TILE=16):
  N= 256 : naive    0.029 ms (1137.8 GFLOP/s, ok)  tiled    0.024 ms (1412.4 GFLOP/s, ok)  speedup 1.24x
  N= 512 : naive    0.197 ms (1364.6 GFLOP/s, ok)  tiled    0.153 ms (1757.0 GFLOP/s, ok)  speedup 1.29x
  N=1024 : naive    1.492 ms (1439.2 GFLOP/s, ok)  tiled    1.145 ms (1876.0 GFLOP/s, ok)  speedup 1.30x

Note: the naive kernel's redundant global loads are largely absorbed by the
A4500's L1/L2 caches, so the shared-memory tiling win here is a modest ~1.3x
rather than the textbook ~TILE (=16x) you would see on a cache-starved GPU.
*/

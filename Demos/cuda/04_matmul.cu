/* Square matrix multiply C = A*B, two kernels:
     matmul_naive  — each thread reads A/B straight from global memory
     matmul_tiled  — cooperative tiling through shared memory (the payoff)
   Build: nvcc -arch=sm_70 -o 04_matmul 04_matmul.cu                        */
#include <cstdio>
#include <cstdlib>

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

int main(void) {
    int N = 512;                          /* divisible by TILE (16) */
    size_t bytes = (size_t)N*N*sizeof(float);
    float *A, *B, *C;
    cudaMallocManaged(&A, bytes); cudaMallocManaged(&B, bytes); cudaMallocManaged(&C, bytes);
    for (int i = 0; i < N*N; i++) { A[i] = 1.0f; B[i] = 1.0f; }

    dim3 threads(TILE, TILE), blocks(N/TILE, N/TILE);
    matmul_tiled<<<blocks, threads>>>(A, B, C, N);    /* try matmul_naive too */
    cudaDeviceSynchronize();

    printf("C[0] = %.1f (expect %d)\n", C[0], N);     /* row of ones . col of ones = N */
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}

/* SAXPY with a grid-stride loop (handles any n with any launch config) and
   unified memory (no explicit memcpy).
   Build: nvcc -arch=sm_70 -o 03_saxpy 03_saxpy.cu                          */
#include <cstdio>

__global__ void saxpy(int n, float a, const float *x, float *y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x)        /* grid-stride loop */
        y[i] = a * x[i] + y[i];
}

int main(void) {
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    float *x, *y;
    cudaMallocManaged(&x, bytes);            /* unified memory */
    cudaMallocManaged(&y, bytes);
    for (int i = 0; i < n; i++) { x[i] = 1.0f; y[i] = 2.0f; }

    saxpy<<<256, 256>>>(n, 3.0f, x, y);
    cudaDeviceSynchronize();
    printf("y[0] = %.1f (expect 5.0)\n", y[0]);

    cudaFree(x); cudaFree(y);
    return 0;
}

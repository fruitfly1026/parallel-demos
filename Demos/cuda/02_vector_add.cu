/* Vector add on the GPU: the full host/device dance (malloc, memcpy, launch).
   Build: nvcc -arch=sm_70 -o 02_vector_add 02_vector_add.cu               */
#include <cstdio>
#include <cstdlib>

__global__ void vadd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];          /* guard: n may not divide evenly */
}

int main(void) {
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    float *a = (float*)malloc(bytes), *b = (float*)malloc(bytes), *c = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; }

    float *da, *db, *dc;
    cudaMalloc(&da, bytes); cudaMalloc(&db, bytes); cudaMalloc(&dc, bytes);
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

    int threads = 256, blocks = (n + threads - 1) / threads;
    vadd<<<blocks, threads>>>(da, db, dc, n);

    cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost);
    printf("c[0] = %.1f, c[%d] = %.1f (expect 3.0)\n", c[0], n-1, c[n-1]);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(a); free(b); free(c);
    return 0;
}

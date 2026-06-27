/* Hello, CUDA — launch a kernel and have each thread print its index.
   Build: nvcc -arch=sm_70 -o 01_hello 01_hello.cu                         */
#include <cstdio>

__global__ void hello(void) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;   /* global thread id */
    printf("Hello from thread %d (block %d, lane %d)\n", id, blockIdx.x, threadIdx.x);
}

int main(void) {
    hello<<<2, 4>>>();          /* 2 blocks x 4 threads = 8 threads */
    cudaDeviceSynchronize();    /* wait for the kernel and flush printf */
    return 0;
}

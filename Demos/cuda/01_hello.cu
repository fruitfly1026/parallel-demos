/* Hello, CUDA — launch a kernel so each thread prints its index, then query
   and print the device properties (name, compute capability, SM count, ...)
   through the CUDA runtime API.
   Build: nvcc -O3 -arch=sm_86 -o 01_hello 01_hello.cu                       */
#include <cstdio>
#include <cuda_runtime.h>

/* Tiny helper: abort loudly if a runtime call returns an error. */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err__), __FILE__, __LINE__);           \
            return 1;                                                         \
        }                                                                     \
    } while (0)

__global__ void hello(void) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;   /* global thread id */
    printf("Hello from thread %d (block %d, lane %d)\n", id, blockIdx.x, threadIdx.x);
}

int main(void) {
    /* --- Query the device through the runtime API and report on it. --- */
    int dev = 0, ndev = 0;
    cudaGetDeviceCount(&ndev);
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    printf("=== Device %d of %d ===\n", dev, ndev);
    printf("Name              : %s\n", p.name);
    printf("Compute capability: %d.%d (sm_%d%d)\n", p.major, p.minor, p.major, p.minor);
    printf("SM count          : %d\n", p.multiProcessorCount);
    printf("Max threads/block : %d\n", p.maxThreadsPerBlock);
    printf("Warp size         : %d\n", p.warpSize);
    printf("Global memory     : %.1f GiB\n", p.totalGlobalMem / (1024.0*1024.0*1024.0));
    printf("Shared mem/block  : %zu KiB\n\n", p.sharedMemPerBlock / 1024);

    /* --- Launch a small grid so each thread announces itself. --- */
    hello<<<2, 4>>>();          /* 2 blocks x 4 threads = 8 threads */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());    /* wait for the kernel and flush printf */
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), Xeon w7-2495X 48c:
=== Device 0 of 1 ===
Name              : NVIDIA RTX A4500
Compute capability: 8.6 (sm_86)
SM count          : 56
Max threads/block : 1024
Warp size         : 32
Global memory     : 19.5 GiB
Shared mem/block  : 48 KiB

Hello from thread 0 (block 0, lane 0)
Hello from thread 1 (block 0, lane 1)
Hello from thread 2 (block 0, lane 2)
Hello from thread 3 (block 0, lane 3)
Hello from thread 4 (block 1, lane 0)
Hello from thread 5 (block 1, lane 1)
Hello from thread 6 (block 1, lane 2)
Hello from thread 7 (block 1, lane 3)
*/

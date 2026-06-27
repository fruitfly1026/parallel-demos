# CUDA demos

GPU programming with CUDA C (CSC 548, Topic 6). Run on an ARC **GPU node**.

| File | Shows |
|------|-------|
| `01_hello.cu` | kernel launch, `blockIdx/threadIdx`, launch config `<<<blocks,threads>>>` |
| `02_vector_add.cu` | host/device `cudaMalloc` + `cudaMemcpy` + launch + copy back |
| `03_saxpy.cu` | grid-stride loop + unified memory (`cudaMallocManaged`) |
| `04_matmul.cu` | naive vs **tiled** (shared-memory) matmul + `__syncthreads()` |

## Build & run (ARC)
```bash
module load cuda
make                       # edit -arch=sm_XX in the Makefile for your GPU
srun --gres=gpu:1 ./02_vector_add
```

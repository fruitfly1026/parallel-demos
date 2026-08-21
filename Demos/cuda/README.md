# CUDA demos

GPU programming with CUDA C (CSC 548, Topic 8 — GPU/CUDA). All four were
compiled with `nvcc -O3 -arch=sm_86` and run on a real **NVIDIA RTX A4500**
(Ampere, CUDA 13.0); each `.cu` file ends with an embedded reference-run
comment showing that GPU's actual output.

Every demo includes:
- a **correctness check** — GPU result vs a CPU reference within a float
  tolerance, printing PASS/FAIL and the max error;
- a **parameter sweep** — block dims / grid sizes / matrix sizes — printing how
  timing changes;
- an **inefficient-vs-optimized** kernel pair, timed with `cudaEvent`s, with the
  inefficiency + fix explained in-code.

| File | Shows | **Inefficiency → fix** (timed) | Measured on A4500 |
|------|-------|--------------------------------|-------------------|
| `01_hello.cu` | kernel launch, `blockIdx/threadIdx`, `<<<blocks,threads>>>`, runtime device query | — | A4500, sm_86, 56 SMs |
| `02_vector_add.cu` | `cudaMalloc`/`cudaMemcpy`/launch; block-dim sweep 128/256/1024 | **strided (non-coalesced)** global access — a warp's 32 loads scatter across memory → **coalesced** access (consecutive threads → consecutive addresses = one transaction) | **10.6×** |
| `03_saxpy.cu` | grid sizing & occupancy | **1 block** — 55 of 56 SMs idle → **full grid** (~1792 blocks) that fills the GPU | **57×** |
| `04_matmul.cu` | tiled shared-memory matmul, `__syncthreads` | **naive** matmul re-reads A/B from global memory every step → **shared-memory tiled** (load a tile once, reuse from on-chip) | **1.3×** \* |

\* The tiled win is only ~1.3× here (not the textbook ~16×) because Ampere's
L1/L2 caches already absorb much of the naive kernel's redundant traffic — a
real measured effect, noted in the code rather than hidden.

## Build & run
```bash
export PATH=/usr/local/cuda/bin:$PATH   # this machine; or `module load cuda` on ARC
make                                    # nvcc -O3 -arch=sm_86 (edit -arch for your GPU)
./02_vector_add
./04_matmul
```
On a Slurm GPU node: `srun --gres=gpu:1 ./02_vector_add`.

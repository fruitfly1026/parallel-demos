# CUDA demos

GPU programming with CUDA C (CSC 548, Topic 8 — GPU/CUDA), organised **one
folder per inefficiency pattern**. Each folder holds a matched pair: the
unoptimized version and the fix. Every `.cu` file is standalone — it compiles
and runs on its own, so it can be profiled in isolation.

Every demo includes:
- a **correctness check** — GPU result vs a CPU reference, PASS/FAIL + max error;
- a **parameter sweep** — block dims, grid sizes, matrix sizes, offsets;
- an embedded **reference run** at the bottom of the file, showing that GPU's
  actual output.

The optimized file of every pair opens with a **`WHAT CHANGED vs <naive>`**
block spelling out exactly which lines differ and why.

All numbers below measured on an **NVIDIA RTX A4500** (Ampere sm_86, CUDA 13.0).

| Folder | Pattern | Pair | What changed | Measured |
|--------|---------|------|--------------|----------|
| `strided/` | **Strided access** — one useful word per 32-byte sector, 7/8 of every transaction wasted | `strided_naive.cu` → `strided_opt.cu` | the index expression: scattered permutation → identity mapping. One line. | 3.721 → 0.353 ms — **10.5×** |
| `false_sharing/` | **False sharing** — 8 words of one sector claimed by 8 different warps, so 8 transactions instead of 1 | `false_sharing_naive.cu` → `false_sharing_opt.cu` | the two index lines swapped, so `threadIdx.x` feeds the *contiguous* dimension. | 12.613 → 1.550 ms — **8.14×** |
| `hotspot/` | **Hot spot** — every warp in the block re-reads the same sectors of B from global memory | `hotspot_coalesced.cu` → `hotspot_tiled.cu` | added 32×32×32 shared-memory block tiling. | 1.551 → 1.235 ms — **1.26×** |
| `occupancy/` | **Not a memory pattern** — the GPU is idle, not confused | `saxpy_naive.cu` → `saxpy_opt.cu` | launch configuration only; the kernel is byte-identical. 1 block → 56×32 blocks. | 85.588 → 1.437 ms — **59.6×** |

`01_hello.cu` stays at the top level: kernel launch, `blockIdx`/`threadIdx`,
`<<<blocks,threads>>>`, and a runtime device query.

## Where the code comes from

`false_sharing/` and `hotspot/` use kernels taken verbatim from
[leimao/CUDA-GEMM-Optimization](https://github.com/leimao/CUDA-GEMM-Optimization)
(`gemm_v00`, `gemm_v01`, `gemm_v02`) — the repository cited as reference [20] of
the cuThermo paper and the origin of the paper's Listing 2. Only the host
harnesses are new.

That lets the two GEMM pairs be checked against the paper's own results:

| Step | Paper (Table 4, A4500) | Measured here |
|------|------------------------|---------------|
| `gemm_v00` → `v01` (false sharing) | 721.79% | 713% (8.14×) |
| `gemm_v01` → `v02` (hot spot) | 26.07% | 25.7% (1.26×) |

`strided/` and `occupancy/` are split out of this repo's earlier
`02_vector_add.cu` and `03_saxpy.cu`.

## Build & run

```bash
export PATH=/usr/local/cuda/bin:$PATH   # or `module load cuda` on ARC
make                                    # builds every pair; edit -arch for your GPU
./strided/strided_naive
./strided/strided_opt
```

On a Slurm GPU node: `srun --gres=gpu:1 ./strided/strided_opt`.

`-lineinfo` is on by default in the Makefile — cuThermo and Nsight Compute both
need it to map memory sectors back to source lines.

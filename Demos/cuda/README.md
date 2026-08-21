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

## Memory access patterns

These are the patterns cuThermo's heat map is built to expose.

| Folder | Pattern | Pair | What changed | Measured |
|--------|---------|------|--------------|----------|
| `strided/` | **Strided access** — one useful word per 32-byte sector, 7/8 of every transaction wasted | `strided_naive.cu` → `strided_opt.cu` | the index expression: scattered permutation → identity mapping. One line. | 3.721 → 0.353 ms — **10.5×** |
| `false_sharing/` | **False sharing** — 8 words of one sector claimed by 8 different warps, so 8 transactions instead of 1 | `false_sharing_naive.cu` → `false_sharing_opt.cu` | the two index lines swapped, so `threadIdx.x` feeds the *contiguous* dimension. | 12.613 → 1.550 ms — **8.14×** |
| `hotspot/` | **Hot spot** — every warp in the block re-reads the same sectors of B from global memory | `hotspot_coalesced.cu` → `hotspot_tiled.cu` | added 32×32×32 shared-memory block tiling. | 1.551 → 1.235 ms — **1.26×** |
| `random_hot/` | **Random hot spot** — input-dependent gather, no structure to tile | `random_hot_naive.cu` → `random_hot_opt.cu` | the **data**, not the kernel: nonzero columns banded instead of scattered. | 4.167 → 0.887 ms — **4.70×** |
| `smem_abuse/` | **Abuse of shared memory** — a per-thread vector parked in SMEM that no other thread ever reads | `smem_abuse_naive.cu` → `smem_abuse_opt.cu` | accumulator moved from `__shared__` to registers; 16 KB/block returned to L1. | 0.0745 → 0.0662 ms — **1.13×** |
| `misalignment/` | **Misalignment** — a warp's 128 bytes straddling 5 sectors instead of 4 | `misalignment_naive.cu` → `misalignment_opt.cu` | row pitch padded 1001 → 1008 floats. | sectors/request 4.28 → 3.43 (**−24.6%**), but **no change in time** — see below |

## Other GPU inefficiencies

Not memory-access patterns, and deliberately included for contrast: a memory
heat map is blind to all three.

| Folder | Pattern | Pair | What changed | Measured |
|--------|---------|------|--------------|----------|
| `thread_divergence/` | **Warp divergence** — a branch splitting a warp, so both paths run with lanes masked off | `thread_divergence_naive.cu` → `thread_divergence_opt.cu` | reduction loop reversed: `tid % (2*s) == 0` → `tid < s`, so active lanes are a contiguous prefix. | 0.5030 → 0.3472 ms — **1.45×** (1.39× from divergence alone) |
| `bank_conflict/` | **Shared-memory bank conflicts** — 32 lanes hitting one of 32 banks, serialising 32 ways | `bank_conflict_naive.cu` → `bank_conflict_opt.cu` | one character: `tile[32][32]` → `tile[32][33]`. | 4,063,232 → **0** conflicts; **1.87×** at 512², 1.07× at 2048² |
| `low_occupancy/` | **Low occupancy** — the GPU is idle, not confused | `saxpy_naive.cu` → `saxpy_opt.cu` | launch configuration only; the kernel is byte-identical. 1 block → 56×32 blocks. | 85.588 → 1.437 ms — **59.6×** |

`01_hello.cu` stays at the top level: kernel launch, `blockIdx`/`threadIdx`,
`<<<blocks,threads>>>`, and a runtime device query.

## Two demos that deliberately show a *negative* result

Worth reading before trusting any profiler output:

- **`misalignment/`** shows a real pattern — 24.6% more sector transactions —
  that costs **zero** wall-clock time, because misalignment adds L1↔L2 sector
  traffic but not DRAM traffic, and the kernel is DRAM-bandwidth-bound. The
  cuThermo paper's own numbers agree: SpMV is its misalignment case and has the
  smallest improvement in Table 4 (1.85%).
- **`random_hot/`** includes a shared-memory staging kernel that is correct and
  **0.88×** — slower. Once the data is reordered, L1 already serves the working
  set, and the staging pass is pure overhead. "Put it in shared memory" is a
  hypothesis, not a fix.

Both are the same lesson: *know what your kernel is bound by before deciding a
pattern is worth fixing.* `bank_conflict/` makes it quantitative — the identical
optimization is worth 1.87× at a cache-resident size and 1.07× at a DRAM-bound
one.

## Citations — where the code comes from

**GEMM / matmul (`false_sharing/`, `hotspot/`).** The kernels `gemm_v00`,
`gemm_v01` and `gemm_v02` are **adopted from Lei Mao's
[CUDA-GEMM-Optimization](https://github.com/leimao/CUDA-GEMM-Optimization)**
(`src/00_non_coalesced_global_memory_access.cu`,
`src/01_coalesced_global_memory_access.cu`, `src/02_2d_block_tiling.cu`).
That repository is cited as reference [20] of the cuThermo paper and is the
origin of the paper's Listing 2. The kernels here are taken verbatim; only the
host harnesses are ours. `hotspot_tiled.cu` additionally inlines leimao's
`load_data_from_global_memory_to_shared_memory()` helper from
`include/cuda_gemm_utils.cuh` so the file builds standalone.

Because the kernels are unmodified, the two GEMM pairs can be checked against
the paper's own results:

| Step | Paper (Table 4, A4500) | Measured here |
|------|------------------------|---------------|
| `gemm_v00` → `v01` (false sharing) | 721.79% | 713% (8.14×) |
| `gemm_v01` → `v02` (hot spot) | 26.07% | 25.7% (1.26×) |

**Reduction (`thread_divergence/`).** `reduce0`, `reduce1` and `reduce2` are from
[NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
(`cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu`), i.e. steps 1–3
of Mark Harris's *Optimizing Parallel Reduction in CUDA*. Mechanical edits only:
`cg::sync(cta)` → `__syncthreads()`, and the templated `SharedMemory<T>()`
helper → a plain float array.

**Transpose (`bank_conflict/`).** `transposeCoalesced` and
`transposeNoBankConflicts` from the NVIDIA Technical Blog post *An Efficient
Matrix Transpose in CUDA C/C++*.

**Offset/stride microbenchmarks (`misalignment/`).** The `offset` kernel from the
NVIDIA Technical Blog post *How to Access Global Memory Efficiently in CUDA
C/C++ Kernels*, also in the CUDA C++ Best Practices Guide.

**`strided/` and `low_occupancy/`** are split out of this repo's earlier
`02_vector_add.cu` and `03_saxpy.cu`.

**`smem_abuse/` and `random_hot/`** are ours. The cuThermo paper's cases for these
patterns (PASTA's `spt_TTMRankRBNnzKernelSM`, cuSZp's compress kernels, SpMV)
cannot be lifted out of their applications, so these are self-contained kernels
written to reproduce the same heat-map signature. The pattern is faithful; the
speedups are these benchmarks' own, not the paper's.

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

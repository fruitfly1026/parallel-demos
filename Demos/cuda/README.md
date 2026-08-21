# CUDA demos

GPU programming with CUDA C (CSC 548, Topic 8 — GPU/CUDA), organised **one
folder per inefficiency pattern**. Each folder holds a matched pair named after
the problem it solves — `gemm`, `vadd`, `spmv`, `saxpy`, … — with a `_naive`
and an `_opt` version. Every `.cu` file is standalone: it compiles and runs on
its own, so it can be profiled in isolation.

Every demo includes:
- a **correctness check** — GPU result vs a CPU reference, PASS/FAIL + max error;
- a **parameter sweep** — block dims, grid sizes, matrix sizes, offsets;
- an embedded **reference run** at the bottom of the file, showing that GPU's
  actual output.

The optimized file of every pair opens with a **`WHAT CHANGED vs <naive>`**
block spelling out exactly which lines differ and why.

All numbers below measured on an **NVIDIA RTX A4500** (Ampere sm_86, CUDA 13.0).

## Memory access patterns

These are the patterns exposed by the heat map of **cuThermo** [[1]](#ref1).

| Folder | Pattern | Pair | What changed | Measured |
|--------|---------|------|--------------|----------|
| `strided/` | **Strided access** — one useful word per 32-byte sector, 7/8 of every transaction wasted | `vadd_naive.cu` → `vadd_opt.cu` | the index expression: scattered permutation → identity mapping. One line. | 3.721 → 0.353 ms — **10.5×** |
| `false_sharing/` | **False sharing** — words of one sector claimed by different warps, so N transactions instead of 1 | `gemm_naive.cu` → `gemm_opt.cu` | the two index lines swapped, so `threadIdx.x` feeds the *contiguous* dimension. | 12.613 → 1.550 ms — **8.14×** |
| `false_sharing/` | *(same pattern, minimal form)* | `warpsum_naive.cu` → `warpsum_opt.cu` | the 8 per-warp results routed through SMEM so **one** warp writes the sector. | store sectors 524,288 → 65,536 — **8×** (time: −6%, see below) |
| `hotspot/` | **Hot spot** — every warp in the block re-reads the same sectors of B from global memory | `gemm_coalesced.cu` → `gemm_tiled.cu` | added 32×32×32 shared-memory block tiling. | 1.551 → 1.235 ms — **1.26×** |
| `random_hot/` | **Random hot spot** — input-dependent gather, no structure to tile | `spmv_naive.cu` → `spmv_opt.cu` | the **data**, not the kernel: nonzero columns banded instead of scattered. | 4.167 → 0.887 ms — **4.70×** |
| `smem_abuse/` | **Abuse of shared memory** — a per-thread vector parked in SMEM that no other thread ever reads | `ttm_naive.cu` → `ttm_opt.cu` | accumulator moved from `__shared__` to registers; 16 KB/block returned to L1. | 0.0745 → 0.0662 ms — **1.13×** |
| `misalignment/` | **Misalignment** — a warp's 128 bytes straddling 5 sectors instead of 4 | `rowscale_naive.cu` → `rowscale_opt.cu` | row pitch padded 1001 → 1008 floats. | sectors/request 4.28 → 3.43 (**−24.6%**), time unchanged |

## Other GPU inefficiencies

Not memory-access patterns, and deliberately included for contrast: a memory
heat map is blind to all three.

| Folder | Pattern | Pair | What changed | Measured |
|--------|---------|------|--------------|----------|
| `thread_divergence/` | **Warp divergence** — a branch splitting a warp, so both paths run with lanes masked off | `reduction_naive.cu` → `reduction_opt.cu` | reduction loop reversed: `tid % (2*s) == 0` → `tid < s`, so active lanes are a contiguous prefix. | 0.5030 → 0.3472 ms — **1.45×** (1.39× from divergence alone) |
| `bank_conflict/` | **Shared-memory bank conflicts** — 32 lanes hitting one of 32 banks, serialising 32 ways | `transpose_naive.cu` → `transpose_opt.cu` | one character: `tile[32][32]` → `tile[32][33]`. | 4,063,232 → **0** conflicts; **1.87×** at 512², 1.07× at 2048² |
| `low_occupancy/` | **Low occupancy** — the GPU is idle, not confused | `saxpy_naive.cu` → `saxpy_opt.cu` | launch configuration only; the kernel is byte-identical. 1 block → 56×32 blocks. | 85.588 → 1.437 ms — **59.6×** |

`01_hello.cu` stays at the top level: kernel launch, `blockIdx`/`threadIdx`,
`<<<blocks,threads>>>`, and a runtime device query.

## Three demos that deliberately record a *negative* result

Worth reading before trusting any profiler output. In each, the pattern is
real and the metric moves exactly as predicted — and the clock does not.

- **`misalignment/`** — 24.6% fewer sector transactions, **zero** time saved.
  Misalignment adds L1↔L2 traffic but not DRAM traffic, and the kernel is
  DRAM-bandwidth-bound. The cuThermo paper agrees: SpMV is its misalignment
  case and has the smallest improvement in its Table 4 (1.85%).
- **`false_sharing/warpsum_*`** — store sectors drop by exactly 8×, and the
  "fixed" kernel is 6% **slower**. A reduction reads 64 MB and writes 2 MB, so
  removing 7/8 of a 3% slice cannot outrun the added `__syncthreads()`. The
  `gemm_*` pair in the same folder shows the identical pattern worth 8.14×
  when it sits on the critical path.
- **`random_hot/`** — includes a shared-memory staging kernel that is correct
  and **0.88×**, i.e. slower. Once the data is reordered, L1 already serves the
  working set and the staging pass is pure overhead.

`bank_conflict/` makes the point quantitative: the identical optimization is
worth 1.87× at 512×512 and 1.07× at 2048×2048. **Know what your kernel is
bound by before deciding a pattern is worth fixing.**

## Verification

Every kernel was compiled and run on the A4500. Beyond the default
configuration each pair was rebuilt and re-run across a parameter sweep —
problem sizes, block dimensions, tile/rank/bandwidth parameters, and sizes that
do *not* divide the block size — and checked under `compute-sanitizer`:

- 20/20 demos: **0 errors** under `compute-sanitizer --tool memcheck`;
- GEMM verified at N = 128, 256, 512, 1000, 1024, 1536 (including sizes that
  are not multiples of the 32×32 block);
- reduction at n = 255, 2²⁰, 1,000,003, 2²⁴ and blockDim 64–1024;
- transpose at 256²–4096²; SpMV at m = 2¹⁸–2²², 4–32 nnz/row, band 16–300;
- TTM at RANK 4–32, NNZ 8–64, and n not a multiple of the block size.

Two size preconditions are enforced at run time rather than left implicit:
`vadd_*` needs `n % STRIDE == 0` (the scattered index is a permutation only
then) and `transpose_*` needs `width % TILE == 0`. Both now print an error and
exit rather than silently computing a wrong answer.

## Citations — where the code comes from

<a name="ref1"></a>**[1]** Yanbo Zhao, Jinku Cui, Zecheng Li, Shuyin Jiao, Xu Liu,
and Jiajia Li. *cuThermo: Understanding GPU Memory Inefficiencies with Heat Map
Profiling*. arXiv preprint [arXiv:2507.18729](https://arxiv.org/abs/2507.18729),
2025.

**GEMM (`false_sharing/gemm_*`, `hotspot/gemm_*`).** The kernels `gemm_v00`,
`gemm_v01` and `gemm_v02` are **adopted from Lei Mao's
[CUDA-GEMM-Optimization](https://github.com/leimao/CUDA-GEMM-Optimization)**
(`src/00_non_coalesced_global_memory_access.cu`,
`src/01_coalesced_global_memory_access.cu`, `src/02_2d_block_tiling.cu`).
That repository is cited as reference [20] of [[1]](#ref1) and is the origin of
that paper's Listing 2. The kernels here are taken verbatim; only the host
harnesses are ours. `hotspot/gemm_tiled.cu` additionally inlines leimao's
`load_data_from_global_memory_to_shared_memory()` helper from
`include/cuda_gemm_utils.cuh` so the file builds standalone.

Because the kernels are unmodified, these two pairs can be checked against the
paper's own results — same GPU, same kernels:

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

**Offset microbenchmark (`misalignment/`).** The `offset` kernel from the NVIDIA
Technical Blog post *How to Access Global Memory Efficiently in CUDA C/C++
Kernels*, also in the CUDA C++ Best Practices Guide.

**`strided/` and `low_occupancy/`** are split out of this repo's earlier
`02_vector_add.cu` and `03_saxpy.cu`.

**`smem_abuse/`, `random_hot/` and `false_sharing/warpsum_*` are ours.** The
paper's cases for these patterns (PASTA's `spt_TTMRankRBNnzKernelSM`, cuSZp's
compress kernels, SpMV on a SuiteSparse matrix) cannot be lifted out of their
applications, so these are self-contained kernels written to reproduce the same
heat-map signature. **The pattern is faithful; the speedups are these
benchmarks' own and are not comparable to the paper's.**

## Build & run

```bash
export PATH=/usr/local/cuda/bin:$PATH   # or `module load cuda` on ARC
make                                    # builds every pair; edit -arch for your GPU
./false_sharing/gemm_naive
./false_sharing/gemm_opt
```

On a Slurm GPU node: `srun --gres=gpu:1 ./false_sharing/gemm_opt`.

`-lineinfo` is on by default in the Makefile — cuThermo and Nsight Compute both
need it to map memory sectors back to source lines.

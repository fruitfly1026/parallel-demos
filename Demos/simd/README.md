# SIMD demos (x86 AVX2 / AVX-512)

Data-level parallelism with Intel intrinsics and compiler auto-vectorization
(CSC 548, Topic 5 — SIMD).
**Requires an x86 CPU with AVX2 + FMA** (AVX-512 is used when available; not
Apple-silicon laptops). All four demos were compiled and run on a real
**Intel Xeon w7-2495X** (AVX-512F/BW/VL/DQ…), gcc 13.3, `-O3 -march=native`.
Each `.c` file ends with an embedded reference-run comment showing that
machine's actual output.

Every demo includes:
- a **correctness check** — scalar (or double-precision) reference vs the SIMD
  result, compared within tolerance, printing PASS/FAIL and the max/rel error;
- a **size sweep** — N = 2^10, 2^20, 2^26 — so you can watch the scalar→SIMD
  gap change as the working set outgrows the caches and hits DRAM bandwidth;
- an **inefficient-vs-optimized** timed comparison with the fix explained in a
  code comment, printing both times and the speedup.

| File | Shows | Inefficient → optimized |
|------|-------|--------------------------|
| `01_vector_add.c` | `_mm256_add_ps` / `_mm512_add_ps`, 8/16 floats per iter | scalar → AVX2 → AVX-512; aligned vs unaligned loads |
| `02_saxpy.c` | `y=a*x+y` with fused multiply-add | scalar → AVX2 mul+add (2 ops) → AVX2 FMA (1 op) → AVX-512 FMA |
| `03_dot_product.c` | FMA reduction + horizontal lane sum | scalar → AVX2 1 accumulator (latency-bound) → 4 accumulators (ILP) → AVX-512 |
| `04_auto_vectorize.c` | compiler auto-vectorization (`restrict`, `ivdep`) | `-fno-tree-vectorize` → `-O3 -march=native` |

Vector add / SAXPY / dot product are all **memory-bound** (a flop or two per
~12 bytes), so the SIMD speedup is largest while data is in cache and shrinks
toward 1× at N=2^26 where every variant saturates DRAM bandwidth (~19 GB/s
single-thread on this part). The instructive part is *how* the gap moves.

## Build & run
```bash
make            # builds all four + 04_novec, using -O3 -march=native -mavx2 -mfma
make run        # build then run each demo
./01_vector_add

make avx2       # rebuild with the AVX2-only floor (no AVX-512)
make vecreport  # show which loops the compiler vectorized in demo 04
```

# SIMD demos (x86 AVX/AVX2)

Data-level parallelism with Intel intrinsics (CSC 548, Topic 5 — SIMD).
**Requires an x86 CPU with AVX2 + FMA** (the ARC compute nodes; not Apple-silicon laptops).

| File | Shows |
|------|-------|
| `01_vector_add.c` | scalar vs `_mm256_add_ps` (8 floats/iter), timed |
| `02_saxpy.c` | `_mm256_fmadd_ps` (fused multiply-add) + scalar tail |
| `03_dot_product.c` | FMA accumulate + horizontal lane reduction |
| `04_auto_vectorize.c` | `#pragma omp simd` + `restrict`; print the vectorizer report |

## Build & run (ARC)
```bash
module load gcc
make
./01_vector_add
# see which loops vectorized:
gcc -O3 -march=native -fopenmp -fopt-info-vec 04_auto_vectorize.c -o 04_auto_vectorize
```

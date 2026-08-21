# OpenMP demos

Self-contained OpenMP programs (CSC 548, Topic 4 — Shared-Memory Programming).
All five were compiled with `gcc -O3 -fopenmp` and run on a 48-core **Intel
Xeon w7-2495X**; each `.c` file ends with an embedded reference-run comment.

Every demo checks its parallel result against a serial reference (PASS/FAIL),
varies a parameter (thread count / schedule) to show behavioral differences,
and — where relevant — times an inefficient version against the fix.

| File | Shows | **Inefficiency → fix** (timed) | Measured (48 threads) |
|------|-------|--------------------------------|-----------------------|
| `01_hello.c` | parallel region, thread ids, `omp_get_num_threads` vs `omp_get_max_threads` | — | 1 vs 48 threads |
| `02_parallel_for.c` | `parallel for` vector op + correctness | serial → `parallel for` | ~10× (saturates below 48× — the kernel is memory-bandwidth-bound) |
| `03_reduction.c` | `reduction(+:...)` | **shared accumulator with a race** — concurrent `+=` loses updates → **wrong, nondeterministic** result → **`reduction(+:sum)`** (private partials merged at the join) | race sum **WRONG**; reduce **6.2×** and exact |
| `04_schedule.c` | schedules on a load-imbalanced (triangular) loop | **`schedule(static)`** — the high-index thread gets all the heavy iterations and becomes the bottleneck → **`schedule(dynamic)`/`guided`** rebalance at runtime | static **13.8×** / dynamic **17.6×** / guided **18.0×** |
| `05_false_sharing.c` | false sharing — the classic cache trap | **threads write adjacent `double`s on ONE 64-byte cache line** — cores fight over it (coherence ping-pong) → **private accumulator** per thread, or **pad each slot to its own line** | private **5.2×**, padded **1.3×** |

## Build & run
```bash
make                                  # gcc -O3 -fopenmp
OMP_NUM_THREADS=1  ./01_hello
OMP_NUM_THREADS=48 ./01_hello

OMP_NUM_THREADS=48 ./02_parallel_for  # try 1, 4, 48 to watch the speedup grow
OMP_NUM_THREADS=48 ./03_reduction     # rerun a few times: the race result wobbles, reduction is stable
OMP_NUM_THREADS=48 ./04_schedule
OMP_NUM_THREADS=48 ./05_false_sharing
```
`OMP_NUM_THREADS` sets the team size at run time — no recompile needed.

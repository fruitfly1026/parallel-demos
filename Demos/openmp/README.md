# OpenMP demos

Small, self-contained OpenMP programs for in-class demos (shared-memory
threading). Each demo checks its parallel result against a serial reference
(PASS/FAIL), varies a parameter (thread count / schedule) to show behavioral
differences, and — where relevant — times an inefficient version against an
optimized one. Reference runs from a 48-core Intel Xeon w7-2495X are embedded
as comments at the bottom of each `.c` file.

| File | Shows |
|------|-------|
| `01_hello.c` | parallel region, thread ids, `omp_get_num_threads` vs `omp_get_max_threads` |
| `02_parallel_for.c` | `parallel for` vector op + correctness check; serial-vs-parallel speedup |
| `03_reduction.c` | `reduction(+:...)` vs a racing shared accumulator (wrong) — the fix |
| `04_schedule.c` | `static` vs `dynamic` vs `guided` on a load-imbalanced loop |
| `05_false_sharing.c` | false sharing vs private/padded accumulation — the classic cache trap |

## Build & run
```bash
make                                  # gcc -O3 -fopenmp
OMP_NUM_THREADS=1  ./01_hello
OMP_NUM_THREADS=4  ./01_hello
OMP_NUM_THREADS=48 ./01_hello

OMP_NUM_THREADS=48 ./02_parallel_for  # try 1, 4, 48 to see the speedup grow
OMP_NUM_THREADS=48 ./03_reduction     # rerun a few times: the race result wobbles
OMP_NUM_THREADS=48 ./04_schedule
OMP_NUM_THREADS=48 ./05_false_sharing
```
`OMP_NUM_THREADS` sets the team size at run time — no recompile needed.

# parallel-demos — status & reference

Every demo has been **compiled and run on real hardware**, and each carries:
- a **correctness check** vs a reference (serial/CPU) — prints `PASS`/`FAIL`,
- **parameter-varying** runs (sizes / thread counts / schedules / partitions) that print the behavioral difference,
- a deliberately **inefficient vs optimized** version, **timed on the machine**, with the inefficiency + fix explained in-code,
- an embedded `Reference run — …` comment holding the actual machine output.

Where a "textbook" optimization did **not** yield a real win on this single node (CUDA tiling, MPI Isend-overlap, Spark `reduceByKey`), the number is reported **honestly with an in-code explanation** rather than staged.

## Test machine
`jli256-ub01.csc.ncsu.edu` — 48-core **Intel Xeon w7-2495X** (AVX-512), **NVIDIA RTX A4500** (Ampere, sm_86), Ubuntu 24.04, gcc 13.3, OpenJDK 21.

## Toolchains (installed under `/mnt/data/jli/sw`)
| Tool | Version | How to use |
|---|---|---|
| CUDA | 13.0 (system) | `export PATH=/usr/local/cuda/bin:$PATH` → `nvcc -O3 -arch=sm_86` |
| gcc / OpenMP | 13.3 (system) | `gcc -O3 -march=native` (SIMD) · `gcc -O3 -fopenmp` (OpenMP) |
| **OpenMPI** | **4.1.6** (built from source) | `export PATH=/mnt/data/jli/sw/openmpi/bin:$PATH; export LD_LIBRARY_PATH=/mnt/data/jli/sw/openmpi/lib:$LD_LIBRARY_PATH` → `mpicc -O3` / `mpirun --oversubscribe -np N` |
| **Spark** | **4.0.0** (tarball, Java 21) | `export SPARK_HOME=/mnt/data/jli/sw/spark-4.0.0-bin-hadoop3; export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64; export PYSPARK_PYTHON=python3` → `$SPARK_HOME/bin/spark-submit --master local[N] file.py` |

---

## CUDA — RTX A4500, `nvcc 13.0 -arch=sm_86`
| Demo | Added | State | Verified result |
|---|---|---|---|
| `01_hello.cu` | runtime device query, error macro | ✅ runs | A4500, sm_86, 56 SMs |
| `02_vector_add.cu` | CPU-ref check, blockDim sweep (128/256/1024), strided→coalesced | ✅ PASS (0 err) | **10.6×** |
| `03_saxpy.cu` | CPU-ref, grid sweep, 1-block→full-grid | ✅ PASS | **57×** |
| `04_matmul.cu` | CPU-ref both kernels, size sweep (256/512/1024), naive→tiled | ✅ PASS | **1.3×** — Ampere L1/L2 already absorb the naive kernel's redundant loads (documented in-code) |

## SIMD — Xeon w7-2495X AVX-512, `gcc -O3 -march=native`
| Demo | Added | State | Result (in-cache 2¹⁰ → streaming 2²⁶) |
|---|---|---|---|
| `01_vector_add.c` | scalar/AVX2/AVX-512 + aligned loads, size sweep | ✅ PASS | 2.6× → ~1.0× |
| `02_saxpy.c` | scalar/AVX2/FMA/AVX-512 FMA, masked tail | ✅ PASS | up to 7× → 1.08× |
| `03_dot_product.c` | 1-acc → 4-acc (ILP) + accuracy analysis | ✅ PASS | 2.0× → 2.4×, **also far more accurate** (naive fp32 sum saturates ~26% low at 67 M terms) |
| `04_auto_vectorize.c` | `-fno-tree-vectorize` vs `-march=native`, `restrict` | ✅ PASS | 6.4× → 2.6× |

*Teaching point: the SIMD win collapses to ~1× at 2²⁶ where every variant hits the single-thread DRAM bandwidth wall (~19–28 GB/s).*

## OpenMP — 48 cores, `gcc -O3 -fopenmp` (new `Demos/openmp/`)
| Demo | Added | State | Result |
|---|---|---|---|
| `01_hello.c` | parallel region, thread ids @ 1/48 threads | ✅ | — |
| `02_parallel_for.c` | serial-ref check, serial→parallel | ✅ PASS | ~9.7–11.7× (bandwidth-bound) |
| `03_reduction.c` | shared-accumulator race (**FAIL**) vs `reduction(+:)` (**PASS**) | ✅ | race sum wrong; reduce **6.2×** |
| `04_schedule.c` | static / dynamic / guided on a triangular load | ✅ PASS | 13.8× / 17.6× / **18.0×** |
| `05_false_sharing.c` | false-sharing vs private accumulator vs cache-line padded | ✅ PASS | **5.2×** (private), 1.3× (padded) |

## MPI — OpenMPI 4.1.6, `mpicc -O3`
| Demo | Added | State | Result |
|---|---|---|---|
| `01_hello.c` | verify ranks form the set {0..P-1}, np 2/4/8 | ✅ PASS | — |
| `02_send_recv.c` | receiver verifies payload, sizes 1→65536 | ✅ PASS | — |
| `03_ping_pong.c` | N tiny msgs → 1 aggregated msg | ✅ PASS | **172×** |
| `04_collectives.c` | manual O(P) root-loop → `MPI_Bcast`/`Reduce` (O(log P)) | ✅ PASS | 2.7× / 1.75× (grows with P) |
| `05_pi.c` | per-iteration `MPI_Reduce` → one final reduce | ✅ PASS | **82–222×**, err ~1e-13 |
| `06_deadlock.c` | large-msg Send/Send **deadlock** (killed by `timeout 8`) → `Sendrecv` fix | ✅ PASS (fix) | hangs vs completes |

## Spark — Spark 4.0.0 / JDK 21, `spark-submit`
| Demo | Added | State | Result |
|---|---|---|---|
| `01_wordcount.py` | Counter-ref, partition sweep (1/4/8/16), `groupByKey`→`reduceByKey` | ✅ PASS | ~1.1× (single node — honest; the big win needs a real cluster) |
| `02_pi.py` | `math.pi` ref, N & numSlices sweep, per-dart map/reduce → `mapPartitions` | ✅ PASS | error ↓ ~1/√N, `mapPartitions` **2.0×** |
| `03_transformations.py` | lineage/`toDebugString`, recompute→`persist()`, `collect()`-loop→distributed | ✅ PASS | cache **2.64×**, distributed **2.45×** |

---

*Pushed to `fruitfly1026/parallel-demos` (commit `5621b0b`). Reference numbers are single-run best-of-3 on the test machine and will vary with load.*

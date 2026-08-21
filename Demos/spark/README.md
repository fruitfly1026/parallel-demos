# Spark demos (PySpark)

Data-parallel processing with Spark (CSC 548, Topic 10 — MapReduce & Spark).

Each demo is self-contained (it generates its own data — no input file needed)
and prints three things: a **correctness check** against a pure-Python
reference (PASS/FAIL), a **parameter-varying** sweep, and a **timed**
inefficient-vs-optimized comparison with the measured speedup.

| File | Shows | Correctness ref | Timed optimization |
|------|-------|-----------------|--------------------|
| `01_wordcount.py` | `flatMap` -> `map` -> `reduceByKey` (MapReduce pattern) | `collections.Counter` | `groupByKey().mapValues(sum)` vs `reduceByKey(add)` (shuffle volume) |
| `02_pi.py` | Monte-Carlo pi: `parallelize` -> reduction | `math.pi` within tol | per-dart `map().reduce()` vs `mapPartitions().sum()` |
| `03_transformations.py` | lazy transformations vs actions + RDD **lineage** (`toDebugString`) | list comprehension | recompute vs `cache()`, and `collect()`-then-loop vs distributed |

## Run
```bash
module load spark            # on ARC; or `pip install pyspark` locally
spark-submit --master 'local[4]' 01_wordcount.py
spark-submit --master 'local[4]' 01_wordcount.py some_text_file.txt
```
`local[N]` runs Spark on N local cores — perfect for a classroom demo.

## Verified reference results
Measured on Spark 4.0.0 (Scala 2.13, OpenJDK 21), 48-core Xeon w7-2495X,
`--master local[4]`:

| Demo | Correctness | Key measured result |
|------|-------------|---------------------|
| 01 wordcount | PASS (vs Counter) | reduceByKey vs groupByKey ≈ **1.1x** on one local node (grows on a cluster) |
| 02 pi | PASS (err 3e-4 < 0.01) | error shrinks ~1/√N; `mapPartitions` **2.0x** faster than per-dart map |
| 03 transforms | PASS (vs list comp) | `cache()` **2.6x**; distributed vs collect-then-loop **2.5x** |

Each script also embeds these reference numbers as a trailing comment block.

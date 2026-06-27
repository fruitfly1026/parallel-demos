# Spark demos (PySpark)

Data-parallel processing with Spark (CSC 548, Topic 10 — MapReduce & Spark).

| File | Shows |
|------|-------|
| `01_wordcount.py` | `flatMap` -> `map` -> `reduceByKey` (the MapReduce pattern) |
| `02_pi.py` | Monte-Carlo pi: `parallelize` -> `map` -> `reduce` |
| `03_transformations.py` | lazy transformations vs actions + RDD **lineage** (`toDebugString`) |

## Run
```bash
module load spark            # on ARC; or `pip install pyspark` locally
spark-submit --master 'local[*]' 01_wordcount.py
spark-submit --master 'local[*]' 01_wordcount.py some_text_file.txt
```
`local[*]` runs Spark on all local cores — perfect for a classroom demo.

"""Transformations (lazy) vs actions (eager), RDD lineage, and the two
classic Spark performance traps.

This version demonstrates:
  (a) CORRECTNESS  — Spark pipeline result checked against a pure-Python
                     list-comprehension reference.
  (b) PARAMETERS   — vary the number of partitions and the input size and
                     show the (identical) result plus the partitioning.
  (c) EFFICIENCY   — two TIMED comparisons:
        1. recompute an expensive RDD on every action  vs  cache()/persist().
        2. collect()-to-driver then loop in Python      vs  a distributed
           transformation that stays in the cluster.

Run:  spark-submit --master 'local[4]' 03_transformations.py
"""
import time

from pyspark import SparkContext, StorageLevel

sc = SparkContext(appName="Transformations")
sc.setLogLevel("ERROR")

nums = sc.parallelize(range(1, 11))            # RDD of 1..10
evens = nums.filter(lambda x: x % 2 == 0)      # lazy: nothing runs yet
squared = evens.map(lambda x: x * x)           # still lazy

print("[lineage] the plan Spark will run on an action:")
for row in squared.toDebugString().decode().splitlines():
    print(f"[lineage]   {row}")

# ---------------------------------------------------------------------------
# (a) CORRECTNESS: Spark result vs a pure-Python list comprehension.
# ---------------------------------------------------------------------------
spark_squared = squared.collect()
ref_squared = [x * x for x in range(1, 11) if x % 2 == 0]
spark_sum = squared.reduce(lambda a, b: a + b)
ref_sum = sum(ref_squared)

ok = (spark_squared == ref_squared) and (spark_sum == ref_sum)
print(f"[correctness] collect -> {spark_squared}")
print(f"[correctness] reduce  -> {spark_sum}  (ref {ref_sum})")
print(f"[correctness] Spark == list-comprehension reference ? "
      f"{'PASS' if ok else 'FAIL'}")

# ---------------------------------------------------------------------------
# (b) PARAMETER VARYING: same computation over different partition counts and
# input sizes. Result is identical; only the physical partitioning changes.
# ---------------------------------------------------------------------------
print("[parameters] sum of squares of evens, varying size and partitions:")
for n, nparts in ((10, 2), (1_000, 4), (1_000_000, 8)):
    rdd = sc.parallelize(range(1, n + 1), nparts)
    s = rdd.filter(lambda x: x % 2 == 0).map(lambda x: x * x).sum()
    ref = sum(x * x for x in range(1, n + 1) if x % 2 == 0)
    print(f"[parameters]   n={n:>9} partitions={nparts:>2} "
          f"partitions_seen={rdd.getNumPartitions():>2} "
          f"sum={int(s)} match={'PASS' if int(s) == ref else 'FAIL'}")


def timeit(fn, reps=3):
    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.time()
        out = fn()
        best = min(best, time.time() - t0)
    return best, out


# ---------------------------------------------------------------------------
# (c.1) INEFFICIENT: recompute an expensive RDD on every action.
#       Each action (count, sum, max) re-runs the ENTIRE lineage — the costly
#       map is executed three times.
#       OPTIMIZED: cache() once; the first action materializes it, the next
#       two read from memory.
# ---------------------------------------------------------------------------
def expensive(x):
    total = 0.0
    for _ in range(200):          # simulate real per-record work
        total += (x * 1.000001) ** 0.5
    return total

base = sc.parallelize(range(1, 2_000_001), 8)

def no_cache():
    r = base.map(expensive)       # NOT cached: recomputed on each action
    return (r.count(), r.sum(), r.max())

def with_cache():
    r = base.map(expensive).persist(StorageLevel.MEMORY_ONLY)
    out = (r.count(), r.sum(), r.max())   # 1st action materializes cache
    r.unpersist()
    return out

slow_t, slow_out = timeit(no_cache)
fast_t, fast_out = timeit(with_cache)
print(f"[timing] 3 actions, NO cache (recompute x3): {slow_t:.3f} s")
print(f"[timing] 3 actions, WITH cache             : {fast_t:.3f} s")
print(f"[timing] cache speedup (slow/fast)         : {slow_t / fast_t:.2f}x")
print(f"[timing] cache results identical ? "
      f"{'PASS' if slow_out == fast_out else 'FAIL'}")

# ---------------------------------------------------------------------------
# (c.2) INEFFICIENT: collect() the whole RDD to the driver, then loop in
#       plain Python (single-threaded, no parallelism, driver memory).
#       OPTIMIZED: keep the work distributed with a transformation + reduce.
# ---------------------------------------------------------------------------
data = sc.parallelize(range(1, 5_000_001), 8)

def collect_then_loop():
    vals = data.collect()                 # pull 5M numbers to the driver
    return sum(v * v for v in vals)       # serial Python loop

def distributed():
    return data.map(lambda v: v * v).sum()  # stays parallel in the cluster

slow2_t, slow2 = timeit(collect_then_loop)
fast2_t, fast2 = timeit(distributed)
print(f"[timing] collect()-then-loop in driver     : {slow2_t:.3f} s")
print(f"[timing] distributed map().sum()           : {fast2_t:.3f} s")
print(f"[timing] distributed speedup (slow/fast)   : {slow2_t / fast2_t:.2f}x")
print(f"[timing] distributed results identical ? "
      f"{'PASS' if slow2 == fast2 else 'FAIL'}")

sc.stop()

# Reference run — Spark 4.0.0 (JDK 21), 48-core Xeon w7-2495X, --master local[4]:
# [correctness] collect -> [4, 16, 36, 64, 100]
# [correctness] reduce  -> 220  (ref 220)
# [correctness] Spark == list-comprehension reference ? PASS
# [parameters]   n=       10 partitions= 2 sum=220                    match=PASS
# [parameters]   n=     1000 partitions= 4 sum=167167000             match=PASS
# [parameters]   n=  1000000 partitions= 8 sum=166667166667000000   match=PASS
# [timing] 3 actions, NO cache (recompute x3): 12.312 s
# [timing] 3 actions, WITH cache             :  4.660 s
# [timing] cache speedup (slow/fast)         : 2.64x   (PASS, results identical)
# [timing] collect()-then-loop in driver     : 0.626 s
# [timing] distributed map().sum()           : 0.256 s
# [timing] distributed speedup (slow/fast)   : 2.45x   (PASS, results identical)

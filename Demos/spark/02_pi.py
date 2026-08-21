"""Estimate pi by Monte Carlo: throw darts at the unit square, count those
inside the quarter circle. A trivially data-parallel reduction in Spark.

This version demonstrates:
  (a) CORRECTNESS  — estimate compared against math.pi within a tolerance.
  (b) PARAMETERS   — vary numSlices (partitions) and sample size N; print how
                     accuracy improves with N (Monte Carlo error ~ 1/sqrt(N)).
  (c) EFFICIENCY   — per-sample map+reduce (one record per dart, huge shuffle
                     of ints) vs mapPartitions that counts locally and emits
                     one partial sum per partition. Both TIMED, same answer.

Run:  spark-submit --master 'local[4]' 02_pi.py
"""
import math
import random
import time

from pyspark import SparkContext

sc = SparkContext(appName="EstimatePi")
sc.setLogLevel("ERROR")


def inside(_):
    x, y = random.random(), random.random()
    return 1 if x * x + y * y <= 1.0 else 0


def count_partition(idx_iter):
    """Count darts inside the circle for a whole partition; emit ONE int."""
    c = 0
    for _ in idx_iter:
        x, y = random.random(), random.random()
        if x * x + y * y <= 1.0:
            c += 1
    yield c


# ---------------------------------------------------------------------------
# (b) PARAMETER VARYING: accuracy vs sample size N (fixed slices), then vs
# number of slices (fixed N). Monte Carlo error shrinks like 1/sqrt(N).
# ---------------------------------------------------------------------------
print("[parameters] accuracy vs sample size N (numSlices=8):")
for N in (10_000, 100_000, 1_000_000, 10_000_000):
    hits = sc.parallelize(range(N), 8).mapPartitions(count_partition).sum()
    est = 4.0 * hits / N
    print(f"[parameters]   N={N:>10}  pi~={est:.6f}  "
          f"abs_err={abs(est - math.pi):.6f}")

print("[parameters] same N=1,000,000, varying numSlices:")
for slices in (1, 4, 8, 16, 48):
    N = 1_000_000
    hits = sc.parallelize(range(N), slices).mapPartitions(count_partition).sum()
    est = 4.0 * hits / N
    print(f"[parameters]   slices={slices:>2}  pi~={est:.6f}  "
          f"abs_err={abs(est - math.pi):.6f}")

# ---------------------------------------------------------------------------
# (a) CORRECTNESS: a large run must land within tolerance of math.pi.
# ---------------------------------------------------------------------------
N = 10_000_000
hits = sc.parallelize(range(N), 8).mapPartitions(count_partition).sum()
pi_est = 4.0 * hits / N
tol = 0.01
ok = abs(pi_est - math.pi) < tol
print(f"[correctness] N={N} pi~={pi_est:.6f}  math.pi={math.pi:.6f}  "
      f"abs_err={abs(pi_est - math.pi):.6f}  tol={tol}")
print(f"[correctness] within tolerance ? {'PASS' if ok else 'FAIL'}")

# ---------------------------------------------------------------------------
# (c) INEFFICIENT vs OPTIMIZED, TIMED
#
#   INEFFICIENT: parallelize(range(N)).map(inside).reduce(add)
#     Creates one Python object per dart and one map output record per dart;
#     the reduce tree passes N intermediate ints around. Lots of per-record
#     interpreter and serialization overhead.
#
#   OPTIMIZED: mapPartitions(count_partition).sum()
#     Loops in a single Python generator per partition and emits just ONE
#     partial count per partition. Same math, far less overhead. Same answer.
# ---------------------------------------------------------------------------
def timeit(fn, reps=3):
    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.time()
        out = fn()
        best = min(best, time.time() - t0)
    return best, out

M = 5_000_000
slow_t, slow_hits = timeit(
    lambda: sc.parallelize(range(M), 8).map(inside).reduce(lambda a, b: a + b))
fast_t, fast_hits = timeit(
    lambda: sc.parallelize(range(M), 8).mapPartitions(count_partition).sum())

print(f"[timing] map(inside).reduce(add)       (N={M}): {slow_t:.3f} s")
print(f"[timing] mapPartitions(count).sum()    (N={M}): {fast_t:.3f} s")
print(f"[timing] speedup (slow/fast)                 : {slow_t / fast_t:.2f}x")
print(f"[result] pi (optimized path) ~= {4.0 * fast_hits / M:.6f}")

sc.stop()

# Reference run — Spark 4.0.0 (JDK 21), 48-core Xeon w7-2495X, --master local[4]:
# [parameters] accuracy vs sample size N (numSlices=8):
# [parameters]   N=     10000  pi~=3.172000  abs_err=0.030407
# [parameters]   N=    100000  pi~=3.146640  abs_err=0.005047
# [parameters]   N=   1000000  pi~=3.143088  abs_err=0.001495
# [parameters]   N=  10000000  pi~=3.142278  abs_err=0.000685
# [parameters] same N=1,000,000, varying numSlices:
# [parameters]   slices= 1  pi~=3.139148  abs_err=0.002445
# [parameters]   slices= 8  pi~=3.141212  abs_err=0.000381
# [parameters]   slices=48  pi~=3.144272  abs_err=0.002679
# [correctness] N=10000000 pi~=3.142182  math.pi=3.141593  abs_err=0.000590  tol=0.01
# [correctness] within tolerance ? PASS
# [timing] map(inside).reduce(add)    (N=5000000): 0.449 s
# [timing] mapPartitions(count).sum() (N=5000000): 0.222 s
# [timing] speedup (slow/fast)                   : 2.02x
# Accuracy improves ~1/sqrt(N); mapPartitions avoids per-dart record overhead.

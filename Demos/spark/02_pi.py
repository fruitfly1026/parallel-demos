"""Estimate pi by Monte Carlo: throw darts at the unit square, count those
inside the quarter circle. A trivially data-parallel reduction in Spark.
Run:  spark-submit --master 'local[*]' 02_pi.py
"""
import random
from pyspark import SparkContext

sc = SparkContext(appName="EstimatePi")
sc.setLogLevel("WARN")

N = 10_000_000

def inside(_):
    x, y = random.random(), random.random()
    return 1 if x*x + y*y <= 1.0 else 0

count = sc.parallelize(range(N), 8).map(inside).reduce(lambda a, b: a + b)
print(f"Pi is roughly {4.0 * count / N}")

sc.stop()

"""Transformations (lazy) vs actions (eager), and RDD lineage.
Run:  spark-submit --master 'local[*]' 03_transformations.py
"""
from pyspark import SparkContext

sc = SparkContext(appName="Transformations")
sc.setLogLevel("WARN")

nums = sc.parallelize(range(1, 11))           # RDD of 1..10
evens   = nums.filter(lambda x: x % 2 == 0)   # lazy: nothing runs yet
squared = evens.map(lambda x: x * x)          # still lazy

print("Lineage (the plan Spark will run on an action):")
print(squared.toDebugString().decode())

# An ACTION forces the whole chain to execute:
print("collect ->", squared.collect())        # [4, 16, 36, 64, 100]
print("reduce  ->", squared.reduce(lambda a, b: a + b))

sc.stop()

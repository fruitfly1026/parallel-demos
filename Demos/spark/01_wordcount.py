"""Word count — the canonical MapReduce/Spark demo (CSC 548, Topic 10).
Run:  spark-submit --master 'local[*]' 01_wordcount.py [textfile]
"""
import sys
from pyspark import SparkContext

sc = SparkContext(appName="WordCount")
sc.setLogLevel("WARN")

if len(sys.argv) > 1:
    lines = sc.textFile(sys.argv[1])
else:                                   # built-in sample if no file given
    lines = sc.parallelize(["the quick brown fox",
                            "the lazy dog",
                            "the quick dog"])

counts = (lines.flatMap(lambda line: line.split())   # line -> words   (map side)
                .map(lambda w: (w, 1))                # word -> (word,1)
                .reduceByKey(lambda a, b: a + b))     # sum per word    (reduce side)

for word, n in sorted(counts.collect(), key=lambda kv: -kv[1]):
    print(f"{word}\t{n}")

sc.stop()

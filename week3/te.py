from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, CountVectorizer
from pyspark.sql import SparkSession  # <--- cần import

# Tạo SparkSession
spark = SparkSession.builder \
    .appName("Read JSON Example") \
    .getOrCreate()

# Đọc file JSON
df = spark.read.json("datatest.json")

print("Số lượng documents:", df.count())
df.show(5, truncate=80)

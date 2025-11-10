from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("SentimentAnalysis")
    .config("spark.driver.extraJavaOptions", "--enable-preview --add-opens java.base/javax.security.auth=ALL-UNNAMED")
    .config("spark.executor.extraJavaOptions", "--enable-preview --add-opens java.base/javax.security.auth=ALL-UNNAMED")
    .getOrCreate()
)


# Khoi tạo Spark session
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

# data
data = [
    ("I love this movie, fantastic!", 1),
    ("This is terrible and boring", 0),
    ("Absolutely great experience", 1),
    ("Worst film ever", 0),
    ("Highly recommend this one", 1),
    ("So bad, never watch again", 0)
]
columns = ["text", "sentiment"]
df = spark.createDataFrame(data, columns)

# chuyen nhan
df = df.withColumn("label", col("sentiment").cast("double")).drop("sentiment")

# tao pipeline 
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=1000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001,  featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])    

# Huan luyen

model = pipeline.fif(df)

# du doan

predictions = model.transform(df)
evaluator = MulticlassClassificationEvaluator(metricLabel="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Accuracy: {accuracy:.2f}")

spark.stop()
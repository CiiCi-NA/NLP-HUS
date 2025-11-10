import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace
import os

def main():
    # Khởi tạo Spark Session
    spark = SparkSession.builder \
        .appName("Word2VecTraining") \
        .master("local[*]") \
        .getOrCreate()
    
    # Tải tập dữ liệu
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'c4-train.00000-of-01024-30K.json')
    df = spark.read.json(data_path)
    
    # In schema để kiểm tra cấu trúc
    print("Schema of the DataFrame:")
    df.printSchema()
    
    # Tiền xử lý (chỉ thực hiện nếu cột 'text' tồn tại)
    if 'text' in df.columns:
        df_clean = df.select(
            lower(
                regexp_replace(col("text"), r'[^\w\s]', '')  # Loại bỏ dấu câu
            ).alias("text")
        )
    else:
        print("Cột 'text' không tồn tại. Vui lòng kiểm tra cấu trúc JSON.")
        spark.stop()
        return
    
    # Tokenize văn bản
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    df_tokenized = tokenizer.transform(df_clean)
    
    # Cấu hình và huấn luyện mô hình Word2Vec
    word2vec = Word2Vec(
        vectorSize=100,     # Vector 100 chiều
        minCount=5,         # Bỏ qua từ có tần suất < 5
        inputCol="words",   # Cột đầu vào chứa từ tokenized
        outputCol="vector"  # Cột đầu ra chứa vector từ
    )
    model = word2vec.fit(df_tokenized)
    
    # Thể hiện mô hình: Tìm từ đồng nghĩa với 'computer'
    print("5 từ tương tự nhất với 'computer':")
    synonyms = model.findSynonyms("computer", 5)
    synonyms.show()
    
    # Dừng Spark session
    spark.stop()

if __name__ == "__main__":
    main()
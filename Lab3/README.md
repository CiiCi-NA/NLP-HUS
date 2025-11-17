# NLP Lab 4: Word Embeddings

## Giới thiệu

Đây là dự án Lab 4 về Word Embeddings trong khóa học NLP. Dự án bao gồm triển khai các mô hình nhúng từ (word embeddings) sử dụng Gensim và Spark, cùng với trực quan hóa.

## Các bước thực hiện

1. **Task 1: Tải và sử dụng model có sẵn (Gensim)**

   - Tải mô hình pre-trained `glove-wiki-gigaword-50`.
   - Lấy vector từ, tính độ tương đồng, và tìm từ đồng nghĩa.

2. **Task 2: Nhúng câu/văn bản**

   - Triển khai hàm nhúng văn bản bằng trung bình vector từ.

3. **Task 3: Huấn luyện model trên tập dữ liệu nhỏ (Gensim)**

   - Huấn luyện Word2Vec trên `en_ewt-ud-train.txt`.
   - Lưu và tải lại mô hình.

4. **Task 4: Huấn luyện model trên tập dữ liệu lớn (Spark)**

   - Sử dụng Spark để đọc và tiền xử lý `c4-train.00000-of-01024-30K.json`.
   - Huấn luyện Word2Vec với Spark MLlib.

5. **Task 5: Trực quan hóa Embedding**
   - Sử dụng PCA để giảm chiều vector xuống 2D.
   - Vẽ scatter plot để trực quan hóa.

## Hướng dẫn chạy code

1. Kích hoạt môi trường:
   ```bash
   conda activate vnvn
   ```
2. Cài đặt các thư viện cần thiết:
   pip install gensim==4.3.2 pyspark==3.5.0 scikit-learn==1.5.0 matplotlib==3.9.2
3. Chạy các script từ thư mục gốc

- Task 1 & 2: python test/lab4_test.py
- Task 3 :python test/lab4_embedding_training_demo.py
- Task 4 :python test/lab4_spark_word2vec_demo.py

4. Đảm bảo các tệp dữ liệu (en_ewt-ud-train.txt, c4-train.00000-of-01024-30K.json) nằm trong thư mục data

## Phân tích kết quả

Độ tương đồng và từ đồng nghĩa:

- Độ tương đồng giữa 'king' và 'queen' (0.7839) cao, phản ánh mối quan hệ ngữ nghĩa gần gũi. 'computer' có các từ đồng nghĩa như 'software' (0.8815), cho thấy mô hình GloVe nắm bắt tốt ngữ cảnh.

Biểu đồ trực quan hóa:

- Trong scatter plot PCA, 'king' và 'queen' nằm gần nhau, trong khi 'computer' và 'software' tạo cụm riêng, phù hợp với kỳ vọng.

So sánh mô hình:

- Mô hình pre-trained (GloVe) ổn định hơn so với Word2Vec tự huấn luyện, phụ thuộc vào chất lượng dữ liệu en_ewt-ud-train.txt.

Khó khăn và giải pháp

- Lỗi ModuleNotFoundError: Do thiếu thư viện như sklearn hoặc matplotlib. Giải pháp: Cài đặt bằng pip install.
- Lỗi đường dẫn tệp: Tệp c4-train...json không tìm thấy. Giải pháp: Sửa đường dẫn bằng os.path.join.
- Lỗi Spark: Cột text không tồn tại. Giải pháp: Kiểm tra schema và điều chỉnh cột.

Trích dẫn tài liệu

- Gensim: https://radimrehurek.com/gensim/models/word2vec.html
- Spark MLlib: https://spark.apache.org/docs/latest/ml-features.html#word2vec
- Universal Dependencies: https://universaldependencies.org/

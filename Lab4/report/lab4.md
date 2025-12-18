# LAB 4 – TEXT CLASSIFICATION

---

## 1. Giới thiệu

Text Classification là bài toán gán nhãn cho văn bản, trong đó mỗi tài liệu văn bản được gán vào một hoặc nhiều lớp xác định trước. Đây là một trong những bài toán cơ bản và quan trọng trong xử lý ngôn ngữ tự nhiên, với nhiều ứng dụng thực tế như phân tích cảm xúc, phát hiện spam, phân loại chủ đề và lọc nội dung.

Trong Lab 4, một pipeline phân loại văn bản hoàn chỉnh được xây dựng, bắt đầu từ văn bản thô và kết thúc bằng việc huấn luyện và đánh giá một mô hình học máy có giám sát.

---

## 2. Mục tiêu

Mục tiêu của Lab 4 bao gồm:

- Hiểu pipeline cơ bản cho bài toán phân loại văn bản
- Áp dụng tokenization và vectorization đã học ở các lab trước
- Xây dựng và huấn luyện mô hình Logistic Regression
- Đánh giá mô hình bằng các chỉ số phổ biến trong classification

---

## 3. Cơ sở lý thuyết

### 3.1. Text Classification

Text Classification là quá trình ánh xạ một văn bản đầu vào sang một nhãn đầu ra. Trong lab này, bài toán được xây dựng dưới dạng **supervised learning**, trong đó mỗi văn bản huấn luyện đều có nhãn tương ứng.

### 3.2. Pipeline phân loại văn bản

Pipeline tổng quát được sử dụng như sau:

Raw Text → Tokenization → Vectorization → Machine Learning Model → Prediction

- **Tokenization:** Chia văn bản thành các token
- **Vectorization:** Biểu diễn văn bản dưới dạng vector số
- **Model:** Logistic Regression
- **Prediction:** Dự đoán nhãn cho văn bản mới

---

## 4. Chuẩn bị dữ liệu

### 4.1. Dataset

Một tập dữ liệu nhỏ, lưu trong bộ nhớ, được sử dụng để minh họa bài toán phân loại cảm xúc:
"This movie is fantastic and I love it!"
"I hate this film, it's terrible."
"The acting was superb, a truly great experience."
"What a waste of time, absolutely boring."
"Highly recommend this, a masterpiece."
"Could not finish watching, so bad."

Nhãn tương ứng:

- `1`: cảm xúc tích cực
- `0`: cảm xúc tiêu cực

---

## 5. Biểu diễn văn bản

Các văn bản được chuyển đổi sang dạng số bằng **TF-IDF Vectorizer** (hoặc CountVectorizer) đã được xây dựng ở các lab trước.

TF-IDF giúp:

- Giảm trọng số của các từ xuất hiện quá thường xuyên
- Tăng khả năng phân biệt giữa các văn bản

---

## 6. Mô hình phân loại văn bản

### 6.1. Logistic Regression

Logistic Regression là một mô hình tuyến tính phổ biến cho bài toán phân loại nhị phân. Mô hình có ưu điểm:

- Dễ triển khai
- Hiệu quả với tập dữ liệu nhỏ
- Là baseline tốt cho các bài toán NLP

Mô hình được huấn luyện bằng thư viện **scikit-learn**, sử dụng solver `liblinear` phù hợp cho dữ liệu nhỏ.

---

### 6.2. Huấn luyện và dự đoán

- Dữ liệu được chia thành tập huấn luyện và tập kiểm tra theo tỷ lệ 80/20
- Mô hình được huấn luyện trên tập huấn luyện
- Dự đoán được thực hiện trên tập kiểm tra

---

## 7. Đánh giá mô hình

Mô hình được đánh giá bằng các chỉ số phổ biến trong classification:

- **Accuracy:** Tỷ lệ dự đoán đúng trên tổng số mẫu
- **Precision:** Độ chính xác của các dự đoán dương
- **Recall:** Khả năng phát hiện đúng các mẫu dương
- **F1-score:** Trung bình điều hòa giữa Precision và Recall

### 7.1. Kết quả đánh giá

| Metric    | Giá trị |
| --------- | ------- |
| Accuracy  | 0.50    |
| Precision | 0.50    |
| Recall    | 1.00    |
| F1-score  | 0.67    |

---

### 7.2. Phân tích kết quả

- Recall đạt giá trị cao (1.00), cho thấy mô hình phát hiện được toàn bộ các mẫu dương.
- Precision thấp hơn (0.50), cho thấy mô hình còn dự đoán nhầm nhiều mẫu âm thành dương.
- Accuracy ở mức trung bình do tập dữ liệu nhỏ và mất cân bằng.
- F1-score phản ánh sự đánh đổi giữa Precision và Recall.

---

## 8. Ví dụ nâng cao với PySpark

Đối với các tập dữ liệu lớn không thể xử lý trên một máy đơn, Apache Spark được sử dụng để xây dựng pipeline phân loại văn bản phân tán.

Pipeline trong Spark ML bao gồm:

- Tokenizer
- StopWordsRemover
- HashingTF
- IDF
- LogisticRegression

Toàn bộ pipeline được huấn luyện và đánh giá thông qua Spark ML Pipeline, giúp xử lý dữ liệu lớn hiệu quả.

---

## 9. Hướng cải thiện mô hình

Một số hướng cải thiện hiệu năng mô hình bao gồm:

### 9.1. Cải thiện tiền xử lý

- Loại bỏ nhiễu (HTML, URL, ký tự đặc biệt)
- Giảm kích thước từ vựng
- Điều chỉnh số chiều của vector TF-IDF

### 9.2. Sử dụng embedding nâng cao

- Word2Vec
- Pre-trained embeddings (GloVe, FastText)

### 9.3. Mô hình phức tạp hơn

- Naive Bayes
- Gradient Boosted Trees
- Neural Networks

---

## 10. Kết luận

Lab 4 cung cấp cái nhìn toàn diện về bài toán phân loại văn bản, từ tiền xử lý đến huấn luyện và đánh giá mô hình. Mặc dù Logistic Regression là một mô hình đơn giản, nó đóng vai trò quan trọng như một baseline để so sánh với các mô hình phức tạp hơn trong các nghiên cứu và ứng dụng NLP.

---

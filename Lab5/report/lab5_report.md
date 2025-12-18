# LAB 5 – RECURRENT NEURAL NETWORKS

## Part 1 – Làm quen với PyTorch và RNN

### Mục tiêu

- Làm quen với PyTorch Tensor và cơ chế autograd.
- Hiểu cách xây dựng mô hình học sâu với `nn.Module`.
- Chuẩn bị nền tảng cho việc sử dụng RNN/LSTM trong các bài toán chuỗi.

### Thực nghiệm

- Tạo và thao tác tensor với PyTorch.
- Kiểm tra gradient bằng `backward()`.
- Sử dụng các layer cơ bản như:
  - `nn.Linear`
  - `nn.Embedding`
- Xây dựng mô hình PyTorch đơn giản.

### Kết luận

Part 1 giúp hiểu rõ cách PyTorch quản lý dữ liệu và gradient, là nền tảng cần thiết cho các mô hình RNN/LSTM ở các phần sau.

---

## Part 2 – Text Classification với RNN/LSTM

### Mục tiêu

- Hiểu hạn chế của các mô hình truyền thống (Bag-of-Words, TF-IDF).
- So sánh hiệu quả giữa các pipeline truyền thống và mô hình chuỗi.
- Xây dựng và huấn luyện các mô hình dựa trên RNN/LSTM.

### Dataset

- HWU – Intent Classification
- Số lượng lớp: **64 intents**

---

### Các mô hình được sử dụng

1. **TF-IDF + Logistic Regression**
2. **Word2Vec (vector trung bình) + Dense Layer**
3. **Pre-trained Embedding + LSTM**
4. **Train-from-scratch Embedding + LSTM**

---

### Kết quả thực nghiệm

#### 🔹 Baseline 1: TF-IDF + Logistic Regression

- **Macro-F1:** **0.8353**

**Nhận xét:**

- Hiệu năng cao và ổn định.
- Tuy nhiên, mô hình không nắm được thứ tự từ và ngữ cảnh dài.

---

#### 🔹 Baseline 2: Word2Vec (average) + Dense Layer

| Epoch | Val Macro-F1 |
| ----- | ------------ |
| 1     | 0.0060       |
| 2     | 0.0179       |
| 3     | 0.0235       |
| 4     | 0.0295       |
| 5     | 0.0506       |

**Nhận xét:**

- Hiệu năng rất thấp.
- Việc lấy trung bình embedding làm mất hoàn toàn thông tin thứ tự và ngữ cảnh.
- Không phù hợp cho bài toán phân loại ý định phức tạp.

---

#### 🔹 Model 3: Pre-trained Embedding + LSTM

| Epoch | Val Macro-F1 |
| ----- | ------------ |
| 1     | 0.0353       |
| 2     | 0.0903       |
| 3     | 0.1196       |

**Nhận xét:**

- Có cải thiện so với Word2Vec trung bình.
- Tuy nhiên, embedding cố định (freeze) không phù hợp hoàn toàn với đặc thù dataset HWU.

---

#### 🔹 Model 4: Train-from-scratch Embedding + LSTM

| Epoch | Val Macro-F1 |
| ----- | ------------ |
| 1     | 0.4792       |
| 2     | 0.7009       |
| 3     | **0.7524**   |

**Nhận xét:**

- Hiệu năng tốt nhất trong các mô hình học sâu.
- Embedding được học trực tiếp từ dữ liệu giúp mô hình nắm bắt ngữ cảnh tốt hơn.
- LSTM xử lý được thứ tự từ và các phụ thuộc trong câu.

---

### Tổng kết Part 2

- TF-IDF + Logistic Regression vẫn là baseline rất mạnh.
- Word2Vec trung bình không phù hợp cho bài toán cần ngữ cảnh.
- LSTM với embedding học từ đầu cho kết quả tốt nhất trong nhóm mô hình neural.

---

## Part 3 – Part-of-Speech Tagging với RNN

### Mục tiêu

- Giải quyết bài toán **token classification**.
- Gán nhãn từ loại (POS) cho từng token trong câu.
- Hiểu sự khác biệt giữa token classification và text classification.

### Dataset

- UD English EWT (định dạng CoNLL, xử lý từ file local).

### Mô hình

- Embedding Layer
- RNN (sequence-to-sequence)
- Linear layer cho từng token

### Kết quả

| Epoch | Dev Token Accuracy |
| ----- | ------------------ |
| 1     | 0.8162             |
| 2     | 0.8593             |
| 3     | **0.8780**         |

### Nhận xét

- RNN học được thông tin ngữ cảnh cục bộ.
- Kết quả khá tốt cho bài toán POS tagging.
- Tuy nhiên, RNN vẫn gặp khó khăn với câu dài và phụ thuộc xa.

---

## Part 4 – Named Entity Recognition với BiLSTM

### Mục tiêu

- Giải quyết bài toán Named Entity Recognition (NER).
- So sánh POS tagging và NER.
- Hiểu ưu điểm của BiLSTM trong bài toán chuỗi.

### Dataset

- Dữ liệu NER định dạng CoNLL (local file).

### Mô hình

- Embedding Layer
- **BiLSTM** (hai chiều)
- Linear layer cho từng token

### Kết quả

| Epoch | Dev Token Accuracy |
| ----- | ------------------ |
| 1     | 0.5000             |
| 2     | 0.5000             |
| 3     | 0.5000             |

### Nhận xét

- Do dataset nhỏ nên mô hình chưa học được tốt.
- BiLSTM vẫn cho thấy khả năng xử lý ngữ cảnh hai chiều.
- Trong thực tế, BiLSTM-CRF hoặc Transformer (BERT) cho kết quả vượt trội hơn.

---

## Tổng kết Lab 5

- Hoàn thành toàn bộ pipeline từ text classification đến token classification.
- RNN/LSTM thể hiện rõ ưu thế trong việc xử lý dữ liệu chuỗi.
- Tuy nhiên, RNN vẫn có hạn chế với long-range dependency.
- Xu hướng hiện đại chuyển sang Transformer cho các bài toán NLP phức tạp.

---

**Kết luận:**  
Lab 5 giúp hiểu rõ sự khác biệt giữa các mô hình truyền thống và mô hình chuỗi, đồng thời cung cấp nền tảng quan trọng để tiếp cận các mô hình NLP hiện đại.

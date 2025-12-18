# LAB 6 – TRANSFORMER / BERT

---

## 1. Giới thiệu

Trong Lab 5, các mô hình RNN/LSTM đã được sử dụng để giải quyết các bài toán phân loại văn bản và token classification. Tuy nhiên, các mô hình này vẫn tồn tại hạn chế trong việc xử lý phụ thuộc dài và khó song song hóa khi huấn luyện.

Lab 6 tập trung vào **Transformer**, đặc biệt là **BERT**, một kiến trúc hiện đại dựa trên cơ chế Self-Attention, hiện đang là nền tảng của nhiều hệ thống NLP tiên tiến.

Mục tiêu của Lab 6:

- Hiểu sự khác biệt giữa RNN/LSTM và Transformer.
- Fine-tune mô hình BERT cho các bài toán NLP.
- Áp dụng Transformer cho text classification và token classification.

---

## 2. Tổng quan về Transformer và BERT

### 2.1. Hạn chế của RNN/LSTM

- Xử lý chuỗi theo thứ tự tuần tự, khó song song hóa.
- Khó học các long-range dependencies.
- Hiệu năng giảm với câu dài.

### 2.2. Transformer

Transformer sử dụng **Self-Attention**, cho phép mỗi token trong câu chú ý đến tất cả token còn lại, giúp:

- Nắm bắt ngữ cảnh toàn cục.
- Huấn luyện nhanh hơn nhờ song song hóa.
- Cải thiện hiệu năng trên nhiều bài toán NLP.

### 2.3. BERT

BERT (Bidirectional Encoder Representations from Transformers) là mô hình Transformer encoder được pre-trained trên lượng dữ liệu lớn, học biểu diễn ngữ cảnh hai chiều và có thể fine-tune cho nhiều bài toán khác nhau.

---

## 3. Part 2 – Text Classification với BERT

### 3.1. Bài toán

- **Intent Classification** trên dataset HWU.
- Mỗi câu được gán một nhãn intent tương ứng với hành động của người dùng.

### 3.2. Dataset

- HWU (Human-Computer Interaction intents).
- Số lượng lớp: **64 intents**.
- Dữ liệu được lưu local gồm `train.csv`, `val.csv`, `test.csv`.

---

### 3.3. Mô hình và thiết lập huấn luyện

- Tokenizer: `distilbert-base-uncased`
- Mô hình: `AutoModelForSequenceClassification`
- Optimizer: AdamW
- Learning rate: `2e-5`
- Batch size: 16
- Epochs: 3
- Metric đánh giá: **Macro-F1**

---

### 3.4. Kết quả thực nghiệm

#### Kết quả trên tập validation

| Epoch | Validation Macro-F1 |
| ----- | ------------------- |
| 1     | 0.861               |
| 2     | 0.892               |
| 3     | **0.908**           |

#### Kết quả trên tập test

- **Test Macro-F1:** **0.901**

---

### 3.5. Nhận xét

- BERT đạt hiệu năng cao hơn so với:
  - TF-IDF + Logistic Regression
  - LSTM (Lab 5)
- Mô hình nắm bắt tốt ngữ cảnh toàn cục và quan hệ giữa các từ.
- Không cần thiết kế đặc trưng thủ công như các mô hình truyền thống.

---

## 4. Part 3 – Token Classification (POS Tagging) với BERT

### 4.1. Bài toán

- Gán nhãn từ loại (POS) cho từng token trong câu.
- Đây là bài toán token-level classification.

### 4.2. Dataset

- UD English EWT (định dạng CoNLL).
- Sử dụng file local `en_ewt-ud-train.txt`.
- Chia dữ liệu theo tỷ lệ 90% train – 10% dev.

---

### 4.3. Mô hình và thiết lập huấn luyện

- Tokenizer: `distilbert-base-uncased`
- Mô hình: `AutoModelForTokenClassification`
- Xử lý subword tokenization bằng `is_split_into_words=True`
- Loss: Cross Entropy (bỏ qua token padding)

---

### 4.4. Kết quả thực nghiệm

| Epoch | Dev Token Accuracy |
| ----- | ------------------ |
| 1     | 0.918              |
| 2     | 0.941              |
| 3     | **0.953**          |

---

### 4.5. Nhận xét

- BERT cho kết quả tốt hơn rõ rệt so với RNN trong Lab 5.
- Việc xử lý subword tokenization là bước quan trọng khi dùng Transformer.
- Mô hình học được ngữ cảnh hai chiều hiệu quả.

---

## 5. So sánh RNN/LSTM và Transformer

| Tiêu chí               | RNN / LSTM            | Transformer / BERT |
| ---------------------- | --------------------- | ------------------ |
| Hướng ngữ cảnh         | Một chiều / hai chiều | Hai chiều          |
| Long-range dependency  | Hạn chế               | Rất tốt            |
| Song song hóa          | Kém                   | Tốt                |
| Feature engineering    | Cần                   | Không cần          |
| Hiệu năng NLP hiện đại | Trung bình            | Rất cao            |

---

## 6. Kết luận

- Transformer/BERT vượt trội so với RNN/LSTM trong hầu hết các bài toán NLP.
- Fine-tuning mô hình pre-trained giúp đạt hiệu năng cao với chi phí huấn luyện thấp.
- Lab 6 cung cấp nền tảng quan trọng để tiếp cận các mô hình NLP hiện đại như BERT, RoBERTa và GPT.

---

## 7. Tài liệu tham khảo

1. Vaswani et al., _Attention Is All You Need_, 2017
2. Devlin et al., _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding_, 2018
3. HuggingFace Transformers Documentation

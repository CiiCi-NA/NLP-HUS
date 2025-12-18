# LAB 7 – DEPENDENCY PARSING WITH SPACY

---

## 1. Mục tiêu của Lab

Mục tiêu của Lab 7 là làm quen và thực hành với **Dependency Parsing (phân tích cú pháp phụ thuộc)** nhằm:

- Hiểu cấu trúc ngữ pháp của câu tiếng Anh.
- Xác định quan hệ phụ thuộc giữa các từ trong câu.
- Ứng dụng phân tích cú pháp vào việc trích xuất thông tin.

---

## 2. Giới thiệu Dependency Parsing

Dependency Parsing biểu diễn cấu trúc câu thông qua các quan hệ phụ thuộc giữa các từ:

- **Head**: từ trung tâm
- **Dependent**: từ phụ thuộc

Mỗi câu có một từ gốc (**ROOT**), thường là động từ chính của câu.  
Mỗi token được gán:

- Nhãn phụ thuộc (dependency label)
- Từ head
- Nhãn từ loại (POS)

---

## 3. Công cụ và môi trường thực nghiệm

- Ngôn ngữ lập trình: **Python 3**
- Thư viện NLP: **spaCy**
- Mô hình sử dụng: `en_core_web_sm`
- Môi trường chạy: VS Code trên Windows

Mô hình `en_core_web_sm` được sử dụng do:

- Nhẹ, tải nhanh
- Đủ chính xác cho Dependency Parsing
- Không yêu cầu word embeddings lớn

---

## 4. Phân tích cú pháp phụ thuộc câu đơn

### 4.1. Câu ví dụ

> _The quick brown fox jumps over the lazy dog._

### 4.2. Kết quả phân tích

| Token | Dependency | Head  | POS   |
| ----- | ---------- | ----- | ----- |
| The   | det        | fox   | DET   |
| quick | amod       | fox   | ADJ   |
| brown | amod       | fox   | ADJ   |
| fox   | nsubj      | jumps | NOUN  |
| jumps | ROOT       | jumps | VERB  |
| over  | prep       | jumps | ADP   |
| the   | det        | dog   | DET   |
| lazy  | amod       | dog   | ADJ   |
| dog   | pobj       | over  | NOUN  |
| .     | punct      | jumps | PUNCT |

**ROOT của câu:** `jumps`

### 4.3. Nhận xét

- Động từ **jumps** được xác định là ROOT.
- Danh từ **fox** là chủ ngữ (_nsubj_) của động từ chính.
- Cụm giới từ _over the lazy dog_ bổ nghĩa cho động từ _jumps_.
- Các tính từ được liên kết với danh từ thông qua quan hệ _amod_.

---

## 5. Trực quan hóa cây phụ thuộc

Cây phụ thuộc của câu được trực quan hóa bằng công cụ **displaCy** của spaCy thông qua trình duyệt web.

### Nhận xét

- Cấu trúc cây thể hiện rõ động từ trung tâm và các thành phần phụ thuộc.
- Trực quan hóa giúp dễ hiểu cấu trúc câu hơn so với dạng bảng.
- Công cụ này hữu ích trong giảng dạy và phân tích ngôn ngữ.

---

## 6. Duyệt cây phụ thuộc và phân tích câu phức

### 6.1. Câu ví dụ

> _Apple is looking at buying U.K. startup for $1 billion._

### 6.2. Kết quả phân tích

- **looking** là động từ chính (_ROOT_).
- **Apple** là chủ ngữ (_nsubj_) của động từ _looking_.
- **buying** là thành phần bổ nghĩa cho giới từ _at_.
- **startup** là tân ngữ (_dobj_) của động từ _buying_.
- Cụm _for $1 billion_ thể hiện giá trị giao dịch.

### 6.3. Nhận xét

- Dependency Parsing cho phép xác định rõ các quan hệ ngữ pháp.
- Việc duyệt cây phụ thuộc giúp trích xuất các thành phần quan trọng như:
  - Chủ ngữ
  - Động từ chính
  - Tân ngữ
  - Các cụm giới từ

---

## 7. Trích xuất thông tin từ cây phụ thuộc

### 7.1. Tìm động từ chính

Động từ có nhãn `ROOT` và POS là `VERB` được xác định là động từ chính của câu.

### 7.2. Trích xuất cụm danh từ

Một cụm danh từ bao gồm:

- determiner (_det_)
- adjective (_amod_)
- compound
- danh từ trung tâm

Ví dụ trích xuất được:

- _the big fluffy cat_
- _the small mouse_

### 7.3. Đường đi từ token đến ROOT

Ví dụ với token **mouse**:

Đường đi này cho thấy mối quan hệ phụ thuộc trực tiếp giữa các từ trong câu.

---

## 8. Nhận xét và kết luận

- Dependency Parsing giúp hiểu sâu cấu trúc ngữ pháp của câu.
- spaCy cung cấp API mạnh mẽ, dễ sử dụng và cho kết quả chính xác.
- Việc phân tích và duyệt cây phụ thuộc là nền tảng cho nhiều bài toán NLP nâng cao như:
  - Information Extraction
  - Relation Extraction
  - Question Answering

**Kết luận:**  
Lab 7 giúp nắm vững cách phân tích cú pháp phụ thuộc và ứng dụng thực tế trong xử lý ngôn ngữ tự nhiên.

---

## 9. Tài liệu tham khảo

1. spaCy Documentation
2. Universal Dependencies
3. NLP with Python

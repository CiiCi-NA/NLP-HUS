#  Lab 2: Count Vectorization

*Natural Language Processing (NLP) – Bag-of-Words Representation*

##  Giới thiệu

Lab 2 tập trung vào việc chuyển văn bản thành dạng **vector số** – đây là bước không thể thiếu khi đưa dữ liệu văn bản vào mô hình Machine Learning. Phương pháp được sử dụng là **Bag-of-Words** thông qua việc tự xây dựng một **CountVectorizer**.

Lab này sẽ tái sử dụng **Tokenizer** bạn đã xây dựng ở Lab 1.

---

##  Mục tiêu của Lab

Sau Lab 2, bạn sẽ hiểu và tự cài đặt được:

* Vectorizer interface cho các mô hình biểu diễn văn bản.
* CountVectorizer sử dụng Bag-of-Words.
* Cách xây dựng vocabulary từ corpus.
* Cách biến văn bản thành vector đếm tần suất token.
* Cách kiểm thử vectorizer bằng Tokenizer Lab 1.

---



##  Task 1 — Xây dựng Vectorizer Interface

Trong `src/core/interfaces.py`, tạo abstract class `Vectorizer` với 3 phương thức:

```python
fit(self, corpus: list[str])
transform(self, documents: list[str]) -> list[list[int]]
fit_transform(self, corpus: list[str]) -> list[list[int]]
```

###  Ý nghĩa:

* **fit**: học toàn bộ vocabulary từ corpus.
* **transform**: biến mỗi document thành vector đếm từ.
* **fit_transform**: thực hiện cả hai bước liên tiếp.

---

## 🛠 Task 2 — Cài đặt CountVectorizer

File:Ai


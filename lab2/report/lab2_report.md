# LAB 2 – COUNT VECTORIZATION (BAG-OF-WORDS)


---

## 1. Giới thiệu

Sau bước tiền xử lý văn bản bằng tokenization ở Lab 1, bước tiếp theo trong một pipeline NLP là **biểu diễn văn bản dưới dạng số** để có thể sử dụng cho các mô hình học máy.

Lab 2 tập trung vào phương pháp **Bag-of-Words (BoW)**, một kỹ thuật biểu diễn văn bản cổ điển nhưng rất quan trọng, trong đó mỗi văn bản được biểu diễn bằng một vector đếm số lần xuất hiện của các từ trong một tập từ vựng chung.

---

## 2. Mục tiêu

Mục tiêu của Lab 2 là:

- Hiểu nguyên lý của mô hình Bag-of-Words
- Xây dựng một `CountVectorizer` từ đầu
- Tái sử dụng tokenizer đã xây dựng ở Lab 1
- Chuyển đổi tập văn bản thành **Document-Term Matrix**

---

## 3. Vectorizer Interface

Một interface `Vectorizer` được định nghĩa trong:

# LAB 2 – COUNT VECTORIZATION (BAG-OF-WORDS)

**Môn học:** Xử lý Ngôn ngữ Tự nhiên (NLP)  
**Sinh viên:** _(Điền tên)_  
**Ngày:** _(Điền ngày)_

---

## 1. Giới thiệu

Sau bước tiền xử lý văn bản bằng tokenization ở Lab 1, bước tiếp theo trong một pipeline NLP là **biểu diễn văn bản dưới dạng số** để có thể sử dụng cho các mô hình học máy.

Lab 2 tập trung vào phương pháp **Bag-of-Words (BoW)**, một kỹ thuật biểu diễn văn bản cổ điển nhưng rất quan trọng, trong đó mỗi văn bản được biểu diễn bằng một vector đếm số lần xuất hiện của các từ trong một tập từ vựng chung.

---

## 2. Mục tiêu

Mục tiêu của Lab 2 là:

- Hiểu nguyên lý của mô hình Bag-of-Words
- Xây dựng một `CountVectorizer` từ đầu
- Tái sử dụng tokenizer đã xây dựng ở Lab 1
- Chuyển đổi tập văn bản thành **Document-Term Matrix**

---

## 3. Vectorizer Interface

Một interface `Vectorizer` được định nghĩa trong:
src/core/interfaces.py

Interface này đảm bảo mọi vectorizer đều cung cấp các phương thức:

- `fit(corpus)`
- `transform(documents)`
- `fit_transform(corpus)`

Thiết kế này giúp tách biệt rõ ràng giữa:

- Giai đoạn học từ vựng (fit)
- Giai đoạn chuyển đổi văn bản sang vector (transform)

---

## 4. CountVectorizer

### 4.1. Mô tả

`CountVectorizer` được cài đặt trong:
src/representations/count_vectorizer.py

Vectorizer này:

- Kế thừa từ interface `Vectorizer`
- Nhận vào một `Tokenizer` (từ Lab 1)
- Xây dựng từ vựng (`vocabulary_`) từ tập văn bản huấn luyện
- Biểu diễn mỗi văn bản bằng một vector đếm số lần xuất hiện của các token

---

### 4.2. Cách xây dựng từ vựng

Quá trình xây dựng từ vựng gồm các bước:

1. Duyệt qua từng văn bản trong corpus
2. Tokenize văn bản bằng `RegexTokenizer`
3. Thu thập tất cả token duy nhất
4. Sắp xếp token và gán mỗi token một chỉ số

---

## 5. Thực nghiệm

### 5.1. Corpus thử nghiệm

Tập văn bản thử nghiệm gồm ba câu:
"I love NLP."
"I love programming."
"NLP is a subfield of AI."

Tokenizer được sử dụng là **RegexTokenizer** từ Lab 1.

---

### 5.2. Từ vựng học được

Sau khi áp dụng `fit_transform`, vectorizer học được tập từ vựng sau:

| Index | Token       |
| ----- | ----------- |
| 0     | .           |
| 1     | a           |
| 2     | ai          |
| 3     | i           |
| 4     | is          |
| 5     | love        |
| 6     | nlp         |
| 7     | of          |
| 8     | programming |
| 9     | subfield    |

Tổng số từ trong từ vựng là **10**.

---

### 5.3. Document-Term Matrix

Mỗi văn bản được biểu diễn bằng một vector có độ dài bằng kích thước từ vựng.

**Document 1 – "I love NLP."**
[1, 0, 0, 1, 0, 1, 1, 0, 0, 0]

**Document 2 – "I love programming."**
[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]

**Document 3 – "NLP is a subfield of AI."**
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1]

---

## 6. Phân tích kết quả

- Các vector phản ánh đúng số lần xuất hiện của mỗi token trong văn bản.
- Dấu câu (.) cũng được coi là một token, do RegexTokenizer tách riêng dấu câu.
- Các từ không xuất hiện trong một văn bản có giá trị bằng 0.
- Document-Term Matrix có kích thước **3 × 10** (3 văn bản, 10 token).

---

## 7. Hạn chế của Bag-of-Words

Mặc dù đơn giản và hiệu quả, Bag-of-Words tồn tại nhiều hạn chế:

- Không giữ được thứ tự từ
- Không nắm bắt được ngữ nghĩa
- Vector có thể rất lớn với tập từ vựng lớn
- Không xử lý tốt từ đồng nghĩa

Những hạn chế này là động lực cho các phương pháp biểu diễn nâng cao hơn như TF-IDF, Word2Vec và các mô hình học sâu.

---

## 8. Kết luận

Lab 2 giúp xây dựng nền tảng quan trọng cho việc biểu diễn văn bản trong NLP. Việc tự cài đặt CountVectorizer giúp hiểu rõ:

- Cách xây dựng từ vựng
- Cách tạo Document-Term Matrix
- Các ưu và nhược điểm của mô hình Bag-of-Words

Những kiến thức này là bước đệm cần thiết cho các lab tiếp theo liên quan đến học máy và biểu diễn ngôn ngữ.

---

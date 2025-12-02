#  Lab 1: Text Tokenization

*Natural Language Processing (NLP) – Tokenization Pipeline*

##  Giới thiệu

Tokenization (tách từ) là bước tiền xử lý cơ bản và quan trọng nhất trong NLP. Hầu như mọi pipeline xử lý văn bản — từ thống kê truyền thống đến deep learning — đều bắt đầu bằng việc chuyển văn bản thô thành các đơn vị nhỏ hơn gọi là **tokens**.

Lab 1 trong project này giúp bạn:

* Hiểu cấu trúc một **Tokenizer interface**.
* Tạo **Simple Tokenizer**.
* Tạo **Regex-based Tokenizer** mạnh mẽ hơn.
* Thực nghiệm trên dataset thực tế **UD_English-EWT**.

---



##  Mục tiêu của Lab

###  Task 1 — Simple Tokenizer

* Tạo interface Tokenizer.
* Implement SimpleTokenizer:

  * Chuyển văn bản về lowercase.
  * Tách bằng khoảng trắng.
  * Tách các dấu câu cơ bản (`. , ? !`).

Ví dụ:

```
"<EXAMPLE_TEXT>" → ["<EXAMPLE_TOKEN1>", "<EXAMPLE_TOKEN2>", ...]
```

---

###  Task 2 — Regex-based Tokenizer (Bonus)

Regex mặc định:

```
<YOUR_REGEX>   # ví dụ: \w+|[^\w\s]
```

Tokenizer này xử lý tốt:

* dấu câu dính từ,
* contractions (`isn't`, `I'm`, `let's`),
* số (`123`),
* ký tự đặc biệt.

---

###  Task 3 — Tokenization trên Dataset UD_English-EWT

```
dataset_path = "<PATH_TO_UD_ENGLISH_EWT>"
sample_length = 500
```

---

##  Chạy thử nghiệm

1. Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

2. Chạy file chính:

```bash
python main.py
```

---

##  Kết quả kỳ vọng

Ví dụ với câu:

```
"<EXAMPLE_SENTENCE>"
```

**SimpleTokenizer** → tách cơ bản
**RegexTokenizer** → tách chính xác hơn

---

##  Khó khăn thường gặp & Cách khắc phục

### 1️ Khó khăn với xử lý dấu câu trong SimpleTokenizer

**Vấn đề:** SimpleTokenizer chỉ tách theo khoảng trắng nên không xử lý đúng các trường hợp như:

```
"Hello,world!" → "hello,world!"
```

**Cách khắc phục:**

* Thêm bước replace trước khi split.
* Tuy nhiên RegexTokenizer vẫn là giải pháp đúng đắn hơn.

---

### 2️ Regex khó viết, dễ sai token

**Vấn đề:** Regex không bao quát các trường hợp như:

* dấu nháy đơn `'`
* từ viết tắt (`U.S.`)
* contractions (`we're`)
* ký tự Unicode

**Cách khắc phục:**

* Bắt đầu từ regex đơn giản: `\w+|[^\w\s]`
* Mở rộng dần theo từng lỗi gặp.

---

###  Không load được dataset UD_English-EWT

**Vấn đề:** Sai đường dẫn / file chưa tải / encoding lạ.

**Cách khắc phục:**

* Kiểm tra lại đường dẫn trong code.
* Mở file bằng VS Code để kiểm tra encoding.

---

###  Văn bản thật trong dataset có nhiều ký tự lạ

**Vấn đề:** UD có emoji, Unicode, HTML escaped text (`&amp;`).

**Cách khắc phục:**

* Dùng RegexTokenizer.
* Hoặc mở rộng regex: `r"[A-Za-z0-9]+|[^\w\s]"`.

---

###  Tokenizer không đồng nhất giữa các câu

**Vấn đề:** Một số token đúng ở câu A nhưng sai ở câu B.

**Cách khắc phục:**

* Viết unit test.
* So sánh với tokenizer chuẩn (SpaCy, NLTK).

---

###  Khó phân biệt giữa chữ viết tắt & dấu chấm câu

Ví dụ:

```
"U.S. economy is growing."
```

**Cách khắc phục:**

* Regex cao cấp hơn.
* Rule-based tokenization.
* Chấp nhận sai số nhỏ (vì Lab 1 không yêu cầu hoàn hảo).

---

##  Ý nghĩa của Lab

Bạn sẽ hiểu rõ:

* tokenization hoạt động thế nào,
* vì sao thư viện NLP cần tokenizer phức tạp,
* hạn chế của tokenization thủ công,
* ứng dụng regex trong NLP.

Đây là nền tảng quan trọng cho các lab tiếp theo như:

* xây dựng vocabulary,
* word embeddings,
* mô hình hóa chuỗi.

---

##  Tài liệu tham khảo

* Jurafsky & Martin — *Speech and Language Processing*
* Universal Dependencies: [https://universaldependencies.org/](https://universaldependencies.org/)
* Python `re` documentation
* Các chatbox AI

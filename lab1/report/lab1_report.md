# LAB 1 – TEXT TOKENIZATION

---

## 1. Giới thiệu

Tokenization là một trong những bước tiền xử lý cơ bản và quan trọng nhất trong xử lý ngôn ngữ tự nhiên. Mục tiêu của tokenization là chia văn bản đầu vào thành các đơn vị nhỏ hơn gọi là _token_, thường là từ, số hoặc dấu câu. Chất lượng của bước tokenization ảnh hưởng trực tiếp đến hiệu quả của các bước xử lý NLP tiếp theo như phân tích cú pháp, gán nhãn từ loại hay học biểu diễn ngôn ngữ.

Trong Lab 1, hai phương pháp tokenization được xây dựng và so sánh:

- Tokenizer đơn giản dựa trên quy tắc
- Tokenizer dựa trên biểu thức chính quy (Regex)

---

## 2. Tokenizer đơn giản (Simple Tokenizer)

Tokenizer đơn giản được xây dựng dựa trên các quy tắc cơ bản:

- Chuyển toàn bộ văn bản về chữ thường
- Tách từ dựa trên khoảng trắng
- Xử lý dấu câu ở mức tối thiểu

Phương pháp này có ưu điểm là dễ cài đặt và tốc độ xử lý nhanh. Tuy nhiên, do chỉ dựa trên khoảng trắng, tokenizer này không xử lý tốt các trường hợp dấu câu gắn liền với từ hoặc các cấu trúc phức tạp trong văn bản tự nhiên.

Ví dụ, với câu:

> _Hello, world! This is a test._

Tokenizer đơn giản tạo ra các token vẫn còn dính dấu câu, làm giảm chất lượng biểu diễn văn bản.

---

## 3. Tokenizer dựa trên Regex (Regex Tokenizer)

Tokenizer thứ hai sử dụng biểu thức chính quy để trích xuất token một cách chi tiết hơn. Biểu thức regex cho phép:

- Tách riêng từ, số và dấu câu
- Xử lý văn bản linh hoạt hơn so với tách bằng khoảng trắng

Regex Tokenizer cho kết quả tốt hơn trong việc tách dấu câu và các ký tự đặc biệt. Tuy nhiên, phương pháp này cũng có hạn chế, đặc biệt là khi xử lý các từ viết tắt hoặc từ rút gọn như _isn't_ hay _let's_, vốn bị tách thành nhiều token nhỏ.

---

## 4. Thực nghiệm trên các câu mẫu

Hai tokenizer được đánh giá trên một số câu ví dụ.

### Câu 1

> _Hello, world! This is a test._

- SimpleTokenizer:

['hello,', 'world!', 'this', 'is', 'a', 'test.']

- RegexTokenizer:

['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

RegexTokenizer cho kết quả chính xác hơn khi tách dấu câu.

---

### Câu 2

> _NLP is fascinating... isn't it?_

- SimpleTokenizer:

RegexTokenizer cho kết quả chính xác hơn khi tách dấu câu.

---

### Câu 2

> _NLP is fascinating... isn't it?_

- SimpleTokenizer:
  ['nlp', 'is', 'fascinating...', "isn't", 'it?']

- RegexTokenizer:
  ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

RegexTokenizer xử lý dấu câu tốt hơn nhưng làm vỡ các từ rút gọn.

---

### Câu 3

> _Let's see how it handles 123 numbers and punctuation!_

- SimpleTokenizer:
  ["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation!']

- RegexTokenizer:
  ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']

---

## 5. Thực nghiệm trên tập dữ liệu UD_English-EWT

Để đánh giá tokenizer trên dữ liệu thực tế, một đoạn văn bản từ tập dữ liệu **UD_English-EWT** đã được sử dụng.

Kết quả cho thấy:

- SimpleTokenizer giữ nguyên các chuỗi dài, khó phân tích ở các bước sau
- RegexTokenizer tách chi tiết hơn các thành phần như dấu gạch nối, dấu chấm và số

Điều này cho thấy tokenizer dựa trên regex phù hợp hơn với dữ liệu ngôn ngữ tự nhiên thực tế.

---

## 6. So sánh hai phương pháp

| Tiêu chí             | Simple Tokenizer | Regex Tokenizer |
| -------------------- | ---------------- | --------------- |
| Dễ cài đặt           | Cao              | Trung bình      |
| Xử lý dấu câu        | Kém              | Tốt             |
| Độ chi tiết token    | Thấp             | Cao             |
| Xử lý từ rút gọn     | Tốt              | Kém             |
| Phù hợp dữ liệu thực | Thấp             | Cao             |

---

## 7. Kết luận

Lab 1 giúp làm rõ vai trò quan trọng của tokenization trong các hệ thống NLP. Thông qua việc xây dựng và so sánh hai phương pháp tokenization, có thể thấy rằng:

- Tokenizer đơn giản phù hợp cho các bài toán nhỏ và dữ liệu đơn giản
- Tokenizer dựa trên regex cho kết quả tốt hơn trên dữ liệu thực tế nhưng vẫn còn hạn chế

Trong các hệ thống NLP hiện đại, các phương pháp tokenization nâng cao hơn như subword tokenization được sử dụng để khắc phục các hạn chế này. Lab 1 cung cấp nền tảng cần thiết để tiếp cận các kỹ thuật tokenization hiện đại trong các lab tiếp theo.

---

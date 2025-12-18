# LAB X – TEXT TO SPEECH (TTS)

**Môn học:** Xử lý Ngôn ngữ Tự nhiên (NLP)  
**Nội dung thêm – Tuần 12**  
**Sinh viên:** _(Điền tên)_  
**MSSV:** _(Điền MSSV)_

---

## 1. Bối cảnh và động cơ nghiên cứu

Khả năng tự học đóng vai trò quan trọng đối với sinh viên, đặc biệt là sau khi hoàn thành chương trình đào tạo chính quy. Với sự phát triển mạnh mẽ của Internet, các hệ thống tìm kiếm, các mô hình AI/Agent và nguồn tài nguyên mở, việc tiếp cận tri thức và tự học trở nên dễ dàng hơn bao giờ hết.

Trong bối cảnh đó, Lab X yêu cầu sinh viên đóng vai trò như một nhà nghiên cứu, tìm hiểu tổng quan về bài toán **Text To Speech (TTS)** – một lĩnh vực quan trọng trong xử lý ngôn ngữ và trí tuệ nhân tạo, với nhiều ứng dụng thực tế như:

- Trợ lý ảo
- Hệ thống đọc văn bản
- Công nghệ hỗ trợ người khiếm thị
- Tổng hợp giọng nói cho nội dung số

Mục tiêu của Lab X không phải là triển khai mô hình, mà là **nghiên cứu tổng quan**, đánh giá các hướng tiếp cận, ưu và nhược điểm của từng phương pháp.

---

## 2. Tổng quan bài toán Text To Speech

Text To Speech (TTS) là bài toán chuyển đổi văn bản đầu vào thành tín hiệu giọng nói có thể nghe được, sao cho:

- Nội dung phát âm chính xác
- Ngữ điệu tự nhiên
- Giọng nói dễ nghe và giống con người

Trong lịch sử phát triển, các hệ thống TTS có thể được chia thành **ba cấp độ chính**, phản ánh sự tiến hóa từ các phương pháp dựa trên luật đến các mô hình học sâu hiện đại.

---

## 3. Level 1 – TTS dựa trên luật (Rule-based / Concatenative)

### 3.1. Mô tả

Ở giai đoạn đầu, các hệ thống TTS được xây dựng dựa trên:

- Luật ngữ âm
- Quy tắc âm tiết
- Ghép nối các đơn vị âm thanh (phoneme, syllable)

### 3.2. Ưu điểm

- Chạy rất nhanh
- Tốn ít tài nguyên tính toán
- Dễ áp dụng cho nhiều ngôn ngữ
- Không cần nhiều dữ liệu huấn luyện

### 3.3. Nhược điểm

- Giọng nói thiếu tự nhiên
- Ngữ điệu cứng nhắc
- Khó biểu đạt cảm xúc
- Chất lượng phụ thuộc mạnh vào bộ luật thủ công

### 3.4. Ứng dụng

- Các hệ thống TTS đời đầu
- Thiết bị nhúng, tài nguyên hạn chế

---

## 4. Level 2 – TTS dựa trên Deep Learning

### 4.1. Mô tả

Sự phát triển của Deep Learning đã tạo ra bước ngoặt lớn cho TTS. Các mô hình như:

- Tacotron
- Tacotron 2
- FastSpeech
- VITS

cho phép học trực tiếp mối quan hệ giữa văn bản và đặc trưng âm thanh.

Một hướng tiếp cận phổ biến là xây dựng **pipeline hoàn chỉnh**, trong đó:

- Người dùng tự ghi âm dữ liệu giọng nói
- Mô hình được fine-tune cho từng người dùng
- Mỗi người có một bộ trọng số riêng

### 4.2. Ưu điểm

- Giọng nói tự nhiên hơn rất nhiều so với Level 1
- Kiểm soát tốt hơn về nhịp điệu, cao độ
- Tốn ít tài nguyên hơn so với các mô hình few-shot phức tạp

### 4.3. Nhược điểm

- Cần dữ liệu huấn luyện tương đối lớn cho mỗi giọng
- Khó mở rộng sang đa ngôn ngữ
- Quá trình thu thập dữ liệu gây bất tiện cho người dùng

### 4.4. Ứng dụng

- Hệ thống TTS cá nhân hóa
- Trợ lý ảo có giọng nói cố định

---

## 5. Level 3 – Few-shot / Zero-shot TTS

### 5.1. Mô tả

Level 3 là hướng nghiên cứu hiện đại nhất, cho phép:

- Chỉ cần vài giây âm thanh mẫu
- Tạo giọng nói mang đặc trưng người nói cho trước

Các mô hình tiêu biểu:

- YourTTS
- VALL-E
- Bark
- XTTS

### 5.2. Ưu điểm

- Cực kỳ linh hoạt
- Ít công sức cho người dùng
- Dễ mở rộng sang nhiều giọng nói khác nhau
- Phù hợp cho đa ngôn ngữ

### 5.3. Nhược điểm

- Mô hình rất phức tạp
- Tốn nhiều tài nguyên tính toán
- Khó triển khai trên thiết bị yếu
- Đặt ra nhiều vấn đề đạo đức

---

## 6. So sánh các hướng tiếp cận

| Tiêu chí      | Level 1  | Level 2    | Level 3 |
| ------------- | -------- | ---------- | ------- |
| Tính tự nhiên | Thấp     | Cao        | Rất cao |
| Tài nguyên    | Rất thấp | Trung bình | Cao     |
| Dữ liệu cần   | Không    | Nhiều      | Rất ít  |
| Đa ngôn ngữ   | Tốt      | Khó        | Tốt     |
| Cá nhân hóa   | Không    | Có         | Rất tốt |
| Độ phức tạp   | Thấp     | Trung bình | Rất cao |

---

## 7. Các thách thức nghiên cứu hiện tại

Các hướng nghiên cứu TTS hiện nay tập trung giải quyết những thách thức sau:

- Đảm bảo **hiệu suất nhanh** như các hệ thống truyền thống
- **Giảm tài nguyên tính toán**
- Duy trì **tính tự nhiên của giọng nói**
- Hỗ trợ **đa ngôn ngữ**
- Thể hiện **cảm xúc** trong giọng nói
- Giảm tối đa **công sức của người dùng**

---

## 8. Vấn đề đạo đức trong nghiên cứu TTS

Sự phát triển mạnh mẽ của TTS, đặc biệt là Level 3, đặt ra nhiều vấn đề đạo đức nghiêm trọng:

- Giả mạo giọng nói
- Deepfake âm thanh
- Lạm dụng trong lừa đảo

Một hướng giải quyết quan trọng là:

- **Nhúng watermark** vào âm thanh sinh ra
- Đánh dấu nội dung do AI tạo ra
- Tăng cường kiểm soát và minh bạch trong sử dụng công nghệ

---

## 9. Kết luận

Text To Speech là một lĩnh vực nghiên cứu giàu tiềm năng và đang phát triển nhanh chóng. Mỗi cấp độ tiếp cận (Level 1, 2, 3) đều có ưu và nhược điểm riêng, phù hợp với các nhu cầu và điều kiện tài nguyên khác nhau.

Trong tương lai, các nghiên cứu TTS sẽ tiếp tục hướng tới việc cân bằng giữa:

- Tính tự nhiên
- Hiệu suất
- Tài nguyên
- Tính đạo đức

Lab X giúp sinh viên có cái nhìn toàn cảnh về TTS và chuẩn bị nền tảng cho việc tiếp cận các nghiên cứu chuyên sâu hơn trong lĩnh vực này.

---

## 10. Tài liệu tham khảo

1. Tacotron: Towards End-to-End Speech Synthesis
2. FastSpeech: Fast, Robust and Controllable Text to Speech
3. VALL-E: Neural Codec Language Models
4. spaCy & HuggingFace TTS resources

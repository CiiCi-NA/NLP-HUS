from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os

def main():
    # Đường dẫn đến tập dữ liệu
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'en_ewt-ud-train.txt')
    output_path = "results/word2vec_ewt.model"
    
    # Đọc dữ liệu theo dòng (mỗi dòng là một câu)
    sentences = LineSentence(data_path)
    
    # Huấn luyện mô hình Word2Vec
    model = Word2Vec(
        sentences,
        vector_size=100,  # Vector 100 chiều
        window=5,         # Kích thước cửa sổ ngữ cảnh
        min_count=5,      # Bỏ qua từ có tần suất < 5
        workers=4         # Sử dụng 4 lõi CPU
    )
    
    # Lưu mô hình
    os.makedirs("results", exist_ok=True)
    model.save(output_path)
    
    # Thể hiện cách sử dụng
    print("Các từ tương tự nhất với 'computer':")
    try:
        similar_words = model.wv.most_similar('computer', topn=5)
        for word, similarity in similar_words:
            print(f"{word}: {similarity:.4f}")
    except KeyError:
        print("'computer' không có trong từ điển")
    
    # Thể hiện phép loại suy (king - man + woman ≈ queen)
    try:
        result = model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
        print("\nPhép loại suy: king - man + woman ≈")
        print(f"{result[0][0]}: {result[0][1]:.4f}")
    except KeyError:
        print("Một hoặc nhiều từ không có trong từ điển")

if __name__ == "__main__":
    main()
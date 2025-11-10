import gensim.downloader
import numpy as np
from typing import List, Tuple, Optional
from nltk.tokenize import word_tokenize
import nltk

# Tải dữ liệu NLTK cho tokenization
nltk.download('punkt')

class WordEmbedder:
    def __init__(self, model_name: str):
        """
        Khởi tạo WordEmbedder với mô hình pre-trained.
        
        Args:
            model_name (str): Tên mô hình (ví dụ: 'glove-wiki-gigaword-50')
        """
        self.model = gensim.downloader.load(model_name)
        self.vector_size = self.model.vector_size

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Lấy vector nhúng cho một từ.
        
        Args:
            word (str): Từ cần lấy vector.
            
        Returns:
            Optional[np.ndarray]: Vector nhúng của từ hoặc None nếu từ không có trong từ điển (OOV).
        """
        try:
            return self.model[word]
        except KeyError:
            return None

    def get_similarity(self, word1: str, word2: str) -> float:
        """
        Tính độ tương đồng cosine giữa hai từ.
        
        Args:
            word1 (str): Từ thứ nhất.
            word2 (str): Từ thứ hai.
            
        Returns:
            float: Độ tương đồng cosine. Trả về 0.0 nếu một trong hai từ là OOV.
        """
        try:
            return self.model.similarity(word1, word2)
        except KeyError:
            return 0.0

    def get_most_similar(self, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Tìm top N từ tương tự nhất với từ đã cho.
        
        Args:
            word (str): Từ cần tìm từ tương tự.
            top_n (int): Số lượng từ tương tự cần trả về (mặc định: 10).
            
        Returns:
            List[Tuple[str, float]]: Danh sách các cặp (từ, độ tương đồng). Trả về danh sách rỗng nếu từ là OOV.
        """
        try:
            return self.model.most_similar(word, topn=top_n)
        except KeyError:
            return []

    def embed_document(self, document: str) -> np.ndarray:
        """
        Tạo vector nhúng cho một tài liệu bằng cách lấy trung bình các vector từ.
        
        Args:
            document (str): Tài liệu dạng chuỗi.
            
        Returns:
            np.ndarray: Vector nhúng của tài liệu. Trả về vector 0 nếu không có từ nào được nhận diện.
        """
        # Tokenize tài liệu
        tokens = word_tokenize(document.lower())
        
        # Lấy vector cho các từ được nhận diện
        vectors = [self.get_vector(token) for token in tokens if self.get_vector(token) is not None]
        
        # Nếu không có vector hợp lệ, trả về vector 0
        if not vectors:
            return np.zeros(self.vector_size)
        
        # Tính trung bình các vector
        return np.mean(vectors, axis=0)
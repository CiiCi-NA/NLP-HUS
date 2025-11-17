# C:\Users\CiiCi\NLP\Lab3\test\lab4_test.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representations.word_embedder import WordEmbedder
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():
    # Khởi tạo WordEmbedder
    embedder = WordEmbedder('glove-wiki-gigaword-50')
    
    # Lấy vector cho từ 'king'
    king_vector = embedder.get_vector('king')
    print("Vector cho 'king':")
    print(king_vector[:10], "... (10 chiều đầu tiên)")
    
    # Tính độ tương đồng giữa 'king' và 'queen', 'king' và 'man'
    king_queen_sim = embedder.get_similarity('king', 'queen')
    king_man_sim = embedder.get_similarity('king', 'man')
    print(f"\nĐộ tương đồng giữa 'king' và 'queen': {king_queen_sim:.4f}")
    print(f"Độ tương đồng giữa 'king' và 'man': {king_man_sim:.4f}")
    
    # Lấy 10 từ tương tự nhất với 'computer'
    similar_to_computer = embedder.get_most_similar('computer', top_n=10)
    print("\n10 từ tương tự nhất với 'computer':")
    for word, similarity in similar_to_computer:
        print(f"{word}: {similarity:.4f}")
    
    # Nhúng câu "The queen rules the country."
    sentence = "The queen rules the country."
    doc_vector = embedder.embed_document(sentence)
    print(f"\nVector nhúng cho câu '{sentence}':")
    print(doc_vector[:10], "... (10 chiều đầu tiên)")
    
    # Thêm trực quan hóa với PCA
    words = ['king', 'queen', 'man', 'woman', 'computer', 'software']
    vectors = np.array([embedder.get_vector(word) for word in words])
    
    # Giảm chiều với PCA
    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)
    
    # Vẽ scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(result[:, 0], result[:, 1], c='blue')
    for i, word in enumerate(words):
        plt.annotate(word, (result[i, 0], result[i, 1]))
    plt.title("PCA Visualization of Word Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

if __name__ == "__main__":
    main()
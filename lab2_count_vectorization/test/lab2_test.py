import sys
import os

# Thêm thư mục gốc vào sys.path để đảm bảo import hoạt động
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lab1_tokenization.src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer

def main():
    # Sửa: Khởi tạo RegexTokenizer thay vì Tokenizer trừu tượng
    tokenizer = RegexTokenizer()

    # Instantiate CountVectorizer with the tokenizer
    vectorizer = CountVectorizer(tokenizer)

    # Define a sample corpus
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    # Use fit_transform on the corpus
    document_term_matrix = vectorizer.fit_transform(corpus)

    # Print the learned vocabulary
    print("Learned Vocabulary:")
    for word, index in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]):
        print(f"  {index}: '{word}'")

    # Print the document-term matrix
    print("\nDocument-Term Matrix (Count Vectors):")
    for i, vector in enumerate(document_term_matrix):
        print(f"  Document {i+1}: {vector}")

if __name__ == "__main__":
    main()
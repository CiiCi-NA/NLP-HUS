from lab2_count_vectorization.src.core.interfaces import Vectorizer
from lab1_tokenization.src.core.interfaces import Tokenizer 

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_ = {}  # Dictionary to hold word-to-index mapping

    def fit(self, corpus: list[str]):
        # Step 1: Collect all unique tokens from the corpus
        unique_tokens = set()
        for document in corpus:
            tokens = self.tokenizer.tokenize(document)
            unique_tokens.update(tokens)  # Add all tokens from this document to the set

        # Step 2: Create vocabulary dictionary by sorting the unique tokens and assigning indices
        sorted_tokens = sorted(unique_tokens)  # Sort to ensure consistent ordering
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted_tokens)}

    def transform(self, documents: list[str]) -> list[list[int]]:
        # Check if fit has been called first
        if not self.vocabulary_:
            raise ValueError("Vocabulary not fitted. Call fit() first.")

        # Initialize the list of vectors
        vectors = []
        vocab_size = len(self.vocabulary_)

        for document in documents:
            # Create a zero vector for this document
            vector = [0] * vocab_size
            tokens = self.tokenizer.tokenize(document)

            # Count tokens and update the vector
            for token in tokens:
                if token in self.vocabulary_:
                    index = self.vocabulary_[token]
                    vector[index] += 1

            vectors.append(vector)

        return vectors

    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        self.fit(corpus)
        return self.transform(corpus)
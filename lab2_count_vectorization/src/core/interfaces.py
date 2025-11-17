from abc import ABC, abstractmethod

# ... (Tokenizer interface from Lab 1 remains) ...

class Vectorizer(ABC):
    @abstractmethod
    def fit(self, corpus: list[str]):
        pass

    @abstractmethod
    def transform(self, documents: list[str]) -> list[list[int]]:
        pass

    @abstractmethod
    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        pass
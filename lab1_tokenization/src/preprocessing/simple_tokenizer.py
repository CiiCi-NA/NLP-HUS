from typing import List
from lab1_tokenization.src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = text.split()
        
        return tokens

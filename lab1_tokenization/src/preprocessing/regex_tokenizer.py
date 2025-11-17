import re
from lab1_tokenization.src.core.interfaces import Tokenizer

class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
        tokens = pattern.findall(text)
        tokens = [token.lower() for token in tokens]
        return tokens
import re
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.interfaces import Tokenizer
class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
        tokens = pattern.findall(text)
        tokens = [token.lower() for token in tokens]
        return tokens
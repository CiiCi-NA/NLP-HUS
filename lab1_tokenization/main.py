from lab1_tokenization.src.preprocessing.simple_tokenizer import SimpleTokenizer
from lab1_tokenization.src.preprocessing.regex_tokenizer import RegexTokenizer
from lab1_tokenization.src.core.dataset_loaders import load_raw_text_data

def main():
    # Khởi tạo các tokenizer
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    # Các câu test
    test_sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]

    print("=== Testing Tokenizers on Sample Sentences ===")
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nSentence {i}: '{sentence}'")
        simple_tokens = simple_tokenizer.tokenize(sentence)
        regex_tokens = regex_tokenizer.tokenize(sentence)
        print(f"  SimpleTokenizer: {simple_tokens}")
        print(f"  RegexTokenizer:  {regex_tokens}")


dataset_path = "data/en_ewt-ud-train.txt"  

raw_text = load_raw_text_data(dataset_path)

# Lấy 500 ký tự đầu
sample_text = raw_text[:500]

print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
print(f"Original Sample: {sample_text[:100]}...")

# --- SimpleTokenizer ---
simple_tokenizer = SimpleTokenizer()
simple_tokens = simple_tokenizer.tokenize(sample_text)
print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")

# --- RegexTokenizer ---
regex_tokenizer = RegexTokenizer()
regex_tokens = regex_tokenizer.tokenize(sample_text)
print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")


if __name__ == "__main__":
    main()
import re
 
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # bỏ URL
    text = re.sub(r"[^a-z\s]", "", text)        # bỏ ký tự đặc biệt
    text = re.sub(r"\s+", " ", text).strip()    # bỏ khoảng trắng thừa
    return text

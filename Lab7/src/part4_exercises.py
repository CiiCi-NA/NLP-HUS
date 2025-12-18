import spacy

nlp = spacy.load("en_core_web_md")

# -------- Bài 1: tìm động từ chính --------
def find_main_verb(doc):
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            return token
    return None

# -------- Bài 2: trích xuất cụm danh từ --------
def extract_noun_chunks(doc):
    noun_chunks = []
    for token in doc:
        if token.pos_ == "NOUN":
            chunk = [child.text for child in token.children if child.dep_ in ("det", "amod", "compound")]
            chunk.append(token.text)
            noun_chunks.append(" ".join(chunk))
    return noun_chunks

# -------- Bài 3: đường đi tới ROOT --------
def get_path_to_root(token):
    path = [token.text]
    while token.head != token:
        token = token.head
        path.append(token.text)
    return path


text = "The big fluffy cat chased the small mouse."
doc = nlp(text)

print("Main verb:", find_main_verb(doc))
print("Noun chunks:", extract_noun_chunks(doc))

for token in doc:
    if token.text == "mouse":
        print("Path to root:", get_path_to_root(token))

import spacy

nlp = spacy.load("en_core_web_md")

text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

print(f"{'TEXT':<10} {'DEP':<10} {'HEAD':<10} {'POS':<8}")
print("-" * 40)

for token in doc:
    print(f"{token.text:<10} {token.dep_:<10} {token.head.text:<10} {token.pos_:<8}")

print("\nROOT of sentence:", [t.text for t in doc if t.dep_ == "ROOT"])

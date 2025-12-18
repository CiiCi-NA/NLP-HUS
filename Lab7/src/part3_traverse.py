import spacy

nlp = spacy.load("en_core_web_md")

text = "Apple is looking at buying U.K. startup for $1 billion."
doc = nlp(text)

print(f"{'TEXT':<12} | {'DEP':<10} | {'HEAD':<12} | {'HEAD POS':<8} | CHILDREN")
print("-" * 75)

for token in doc:
    children = [child.text for child in token.children]
    print(f"{token.text:<12} | {token.dep_:<10} | {token.head.text:<12} | {token.head.pos_:<8} | {children}")

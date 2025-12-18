import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_md")

text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

print("Open browser at: http://127.0.0.1:5000")
displacy.serve(doc, style="dep")

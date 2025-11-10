
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.text_classifier import TextClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# --- Dữ liệu mẫu ---
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]

labels = [1, 0, 1, 0, 1, 0]

# --- Chia dữ liệu train/test ---
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.33, random_state=42
)

# --- Tạo vectorizer TF-IDF ---
vectorizer = TfidfVectorizer()

# --- Tạo và huấn luyện classifier ---
clf = TextClassifier(vectorizer)
clf.fit(train_texts, train_labels)

# --- Dự đoán ---
preds = clf.predict(test_texts)

# --- Đánh giá ---
metrics = clf.evaluate(test_labels, preds)

print(" Evaluation Metrics :")
for k, v in metrics.items():
    print(f"{k.capitalize()}: {v:.2f}")

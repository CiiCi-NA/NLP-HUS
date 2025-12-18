
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# gensim
from gensim.models import Word2Vec

# torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# utils
from Lab5.src.common.utils import set_seed, get_device


# ======================================================
# 1. LOAD HWU DATASET
# ======================================================

def load_hwu(data_dir: str):
    
    import os
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    hwu_dir = os.path.join(data_dir, "hwu")

    def read_csv(path):
        df = pd.read_csv(path)

        # case 1: có header
        if "text" in df.columns:
            for lbl in ["intent", "label", "category"]:
                if lbl in df.columns:
                    return df["text"].astype(str), df[lbl].astype(str)

        # case 2: không có header
        df = pd.read_csv(path, header=None)
        if df.shape[1] < 2:
            raise ValueError(f"File {path} không đúng format CSV")

        return df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)

    x_train, y_train = read_csv(os.path.join(hwu_dir, "train.csv"))
    x_val,   y_val   = read_csv(os.path.join(hwu_dir, "val.csv"))
    x_test,  y_test  = read_csv(os.path.join(hwu_dir, "test.csv"))

    le = LabelEncoder()
    le.fit(y_train)

    return (
        x_train.tolist(),
        le.transform(y_train),
        x_val.tolist(),
        le.transform(y_val),
        x_test.tolist(),
        le.transform(y_test),
        le
    )

    """
    Load HWU dataset from:
    Lab5/data/hwu/{train,val,test}.csv
    CSV format: text,intent
    """
    hwu_dir = os.path.join(data_dir, "hwu")

    train_df = pd.read_csv(os.path.join(hwu_dir, "train.csv"))
    val_df   = pd.read_csv(os.path.join(hwu_dir, "val.csv"))
    test_df  = pd.read_csv(os.path.join(hwu_dir, "test.csv"))

    assert "text" in train_df.columns
    assert "intent" in train_df.columns

    le = LabelEncoder()
    le.fit(train_df["intent"])

    x_train = train_df["text"].tolist()
    y_train = le.transform(train_df["intent"])

    x_val = val_df["text"].tolist()
    y_val = le.transform(val_df["intent"])

    x_test = test_df["text"].tolist()
    y_test = le.transform(test_df["intent"])

    return x_train, y_train, x_val, y_val, x_test, y_test, le


# ======================================================
# 2. BASELINE 1: TF-IDF + LOGISTIC REGRESSION
# ======================================================

def baseline_tfidf_lr(x_train, y_train, x_test, y_test):
    print("\n=== Baseline 1: TF-IDF + Logistic Regression ===")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    f1 = f1_score(y_test, preds, average="macro")
    print("Macro-F1:", f1)
    print(classification_report(y_test, preds))

    return f1


# ======================================================
# 3. BASELINE 2: Word2Vec (AVG) + DENSE
# ======================================================

def train_word2vec(sentences, dim=100):
    tokenized = [s.lower().split() for s in sentences]
    return Word2Vec(tokenized, vector_size=dim, window=5, min_count=1, workers=4)


def sentence_avg_vector(text, w2v):
    words = text.lower().split()
    vecs = [w2v.wv[w] for w in words if w in w2v.wv]
    if len(vecs) == 0:
        return np.zeros(w2v.vector_size)
    return np.mean(vecs, axis=0)


class AvgVecDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DenseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def baseline_avg_w2v_dense(x_train, y_train, x_val, y_val, num_classes):
    print("\n=== Baseline 2: Word2Vec (avg) + Dense ===")

    w2v = train_word2vec(x_train)

    X_train = np.stack([sentence_avg_vector(s, w2v) for s in x_train])
    X_val   = np.stack([sentence_avg_vector(s, w2v) for s in x_val])

    train_loader = DataLoader(AvgVecDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader   = DataLoader(AvgVecDataset(X_val, y_val), batch_size=64)

    device = get_device()
    model = DenseClassifier(w2v.vector_size, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds.extend(torch.argmax(model(xb), 1).cpu().numpy())
                golds.extend(yb.numpy())

        f1 = f1_score(golds, preds, average="macro")
        print(f"Epoch {epoch+1} - Val Macro-F1: {f1:.4f}")

    return f1, w2v


# ======================================================
# 4. LSTM MODELS (PRETRAINED vs SCRATCH)
# ======================================================

class Vocab:
    def __init__(self, texts):
        self.PAD = "<PAD>"
        self.UNK = "<UNK>"
        self.itos = [self.PAD, self.UNK]
        for s in texts:
            for w in s.lower().split():
                if w not in self.itos:
                    self.itos.append(w)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text):
        return [self.stoi.get(w, 1) for w in text.lower().split()]


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.X = [vocab.encode(t)[:max_len] for t in texts]
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def collate_fn(batch):
    xs, ys = zip(*batch)
    lens = torch.tensor([len(x) for x in xs])
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = torch.stack(ys)
    return xs, lens, ys


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, emb_weights=None, freeze=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if emb_weights is not None:
            self.embedding.weight.data.copy_(torch.tensor(emb_weights))
        self.embedding.weight.requires_grad = not freeze

        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lens):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


def train_lstm(model, train_loader, val_loader, epochs=3):
    device = get_device()
    model.to(device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xb, lens, yb in train_loader:
            xb, lens, yb = xb.to(device), lens.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, lens), yb)
            loss.backward()
            opt.step()

        model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for xb, lens, yb in val_loader:
                xb, lens = xb.to(device), lens.to(device)
                preds.extend(torch.argmax(model(xb, lens), 1).cpu().numpy())
                golds.extend(yb.numpy())

        f1 = f1_score(golds, preds, average="macro")
        print(f"Epoch {epoch+1} - Val Macro-F1: {f1:.4f}")

    return f1


# ======================================================
# 5. MAIN
# ======================================================

def main():
    set_seed(42)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    x_train, y_train, x_val, y_val, x_test, y_test, le = load_hwu(data_dir)
    num_classes = len(le.classes_)

    # Baseline 1
    baseline_tfidf_lr(x_train, y_train, x_test, y_test)

    # Baseline 2
    _, w2v = baseline_avg_w2v_dense(x_train, y_train, x_val, y_val, num_classes)

    # LSTM models
    vocab = Vocab(x_train)
    train_loader = DataLoader(TextDataset(x_train, y_train, vocab), batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(TextDataset(x_val, y_val, vocab), batch_size=64, collate_fn=collate_fn)

    print("\n=== Model 3: Pretrained Embedding + LSTM ===")
    emb_matrix = np.random.normal(0, 0.1, (len(vocab.itos), w2v.vector_size))
    for w, i in vocab.stoi.items():
        if w in w2v.wv:
            emb_matrix[i] = w2v.wv[w]

    model_pre = LSTMClassifier(len(vocab.itos), w2v.vector_size, 128, num_classes, emb_matrix, freeze=True)
    train_lstm(model_pre, train_loader, val_loader)

    print("\n=== Model 4: Scratch Embedding + LSTM ===")
    model_scratch = LSTMClassifier(len(vocab.itos), 100, 128, num_classes)
    train_lstm(model_scratch, train_loader, val_loader)


if __name__ == "__main__":
    main()

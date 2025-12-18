from __future__ import annotations
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- Utils ----------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load HWU local ----------
def read_hwu_csv(path: str):
    df = pd.read_csv(path)
    # hỗ trợ nhiều format cột
    if "text" in df.columns:
        for lbl in ["intent", "label", "category"]:
            if lbl in df.columns:
                return df["text"].astype(str).tolist(), df[lbl].astype(str).tolist()
    # fallback: không header
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 0].astype(str).tolist(), df.iloc[:, 1].astype(str).tolist()

def load_hwu(data_dir: str):
    hwu_dir = os.path.join(data_dir, "hwu")
    x_train, y_train = read_hwu_csv(os.path.join(hwu_dir, "train.csv"))
    x_val, y_val     = read_hwu_csv(os.path.join(hwu_dir, "val.csv"))
    x_test, y_test   = read_hwu_csv(os.path.join(hwu_dir, "test.csv"))

    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_val   = le.transform(y_val)
    y_test  = le.transform(y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test, le

# ---------- Dataset ----------
class HWUDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            padding=False,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def collate_fn(batch):
    # dynamic padding
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == "labels":
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = torch.nn.utils.rnn.pad_sequence(
                [b[k] for b in batch], batch_first=True, padding_value=0
            )
    # attention mask: 1 for tokens, 0 for padding
    if "attention_mask" not in out:
        out["attention_mask"] = (out["input_ids"] != 0).long()
    return out

# ---------- Train/Eval ----------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_p, all_y = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch["labels"].detach().cpu().numpy()
        logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
        p = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_p.append(p)
        all_y.append(y)
    p = np.concatenate(all_p)
    y = np.concatenate(all_y)
    return f1_score(y, p, average="macro"), classification_report(y, p)

def train():
    set_seed(42)
    device = get_device()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    x_train, y_train, x_val, y_val, x_test, y_test, le = load_hwu(data_dir)
    num_labels = len(le.classes_)

    model_name = "distilbert-base-uncased"  # nhẹ, dễ chạy CPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    train_loader = DataLoader(
        HWUDataset(x_train, y_train, tokenizer, max_len=64),
        batch_size=16, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        HWUDataset(x_val, y_val, tokenizer, max_len=64),
        batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        HWUDataset(x_test, y_test, tokenizer, max_len=64),
        batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(1, 4):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            out = model(**{k: v for k, v in batch.items() if k != "labels"})
            loss = loss_fn(out.logits, batch["labels"])
            loss.backward()
            optim.step()
            total_loss += loss.item()

        val_f1, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | train_loss={total_loss:.2f} | val_macro_f1={val_f1:.4f}")
        best = max(best, val_f1)

    print("\n=== TEST EVALUATION ===")
    test_f1, report = evaluate(model, test_loader, device)
    print("Test macro-F1:", test_f1)
    print(report)

if __name__ == "__main__":
    train()

from __future__ import annotations
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForTokenClassification

PAD_LABEL_ID = -100

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ud_txt(path):
    sents = []
    cur_w, cur_t = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_w:
                    sents.append((cur_w, cur_t))
                    cur_w, cur_t = [], []
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            cur_w.append(parts[1])
            cur_t.append(parts[3])  # UPOS
    if cur_w:
        sents.append((cur_w, cur_t))
    return sents

def build_tag_map(sents):
    tags = sorted({t for _, ts in sents for t in ts})
    tag2id = {t:i for i,t in enumerate(tags)}
    id2tag = {i:t for t,i in tag2id.items()}
    return tag2id, id2tag

class POSBertDataset(Dataset):
    def __init__(self, sents, tokenizer, tag2id, max_len=128):
        self.sents = sents
        self.tok = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len

    def __len__(self): return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx]
        enc = self.tok(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)
        labels = []
        prev = None
        for wi in word_ids:
            if wi is None:
                labels.append(PAD_LABEL_ID)
            elif wi != prev:
                labels.append(self.tag2id[tags[wi]])
            else:
                labels.append(PAD_LABEL_ID)  # chỉ chấm token đầu của word
            prev = wi

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(labels, dtype=torch.long)
        return item

def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        pad_val = 0 if k != "labels" else PAD_LABEL_ID
        out[k] = torch.nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True, padding_value=pad_val)
    if "attention_mask" not in out:
        out["attention_mask"] = (out["input_ids"] != 0).long()
    return out

@torch.no_grad()
def token_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = {k:v.to(device) for k,v in batch.items()}
        logits = model(**{k:v for k,v in batch.items() if k!="labels"}).logits
        pred = torch.argmax(logits, dim=-1)
        y = batch["labels"]
        mask = (y != PAD_LABEL_ID)
        correct += (pred[mask] == y[mask]).sum().item()
        total += mask.sum().item()
    return correct / max(total, 1)

def main():
    set_seed(42)
    device = get_device()

    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "en_ewt-ud-train.txt")
    sents = load_ud_txt(data_path)

    split = int(0.9 * len(sents))
    train_sents = sents[:split]
    dev_sents = sents[split:]

    tag2id, id2tag = build_tag_map(sents)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(tag2id)).to(device)

    train_loader = DataLoader(POSBertDataset(train_sents, tokenizer, tag2id), batch_size=8, shuffle=True, collate_fn=collate_fn)
    dev_loader   = DataLoader(POSBertDataset(dev_sents, tokenizer, tag2id), batch_size=16, shuffle=False, collate_fn=collate_fn)

    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_LABEL_ID)

    for epoch in range(1, 4):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = {k:v.to(device) for k,v in batch.items()}
            optim.zero_grad()
            logits = model(**{k:v for k,v in batch.items() if k!="labels"}).logits
            loss = loss_fn(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
            loss.backward()
            optim.step()
            total_loss += loss.item()
        acc = token_acc(model, dev_loader, device)
        print(f"Epoch {epoch} | loss={total_loss:.2f} | dev_token_acc={acc:.4f}")

if __name__ == "__main__":
    main()



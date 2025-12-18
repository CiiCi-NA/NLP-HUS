from __future__ import annotations
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from Lab5.src.common.utils import set_seed, get_device

PAD_LABEL_ID = -100


# 1. LOAD UD DATA (LOCAL FILE)

def load_ud_file(path):
    
    sentences = []
    current = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            word = parts[1]
            pos = parts[3]   # UPOS
            current.append((word, pos))

    if current:
        sentences.append(current)

    return sentences


# 2. BUILD VOCAB

def build_vocab(sentences):
    word2id = {"<PAD>": 0, "<UNK>": 1}
    tag2id = {"<PAD>": PAD_LABEL_ID}

    for sent in sentences:
        for w, t in sent:
            w = w.lower()
            if w not in word2id:
                word2id[w] = len(word2id)
            if t not in tag2id:
                tag2id[t] = len(tag2id)

    return word2id, tag2id


# 3. DATASET

class POSDataset(Dataset):
    def __init__(self, sentences, word2id, tag2id):
        self.data = sentences
        self.word2id = word2id
        self.tag2id = tag2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        x = [self.word2id.get(w.lower(), 1) for w, _ in sent]
        y = [self.tag2id[t] for _, t in sent]

        return torch.tensor(x), torch.tensor(y)


def collate_fn(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])

    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=PAD_LABEL_ID)

    return xs, lengths, ys


# 4. MODEL

class RNNForPOSTagging(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_tags)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return self.fc(out)


# 5. EVALUATION

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for xb, lengths, yb in loader:
            xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
            logits = model(xb, lengths)
            preds = torch.argmax(logits, dim=-1)

            mask = yb != PAD_LABEL_ID
            correct += (preds[mask] == yb[mask]).sum().item()
            total += mask.sum().item()

    return correct / total


# 6. MAIN
def main():
    set_seed(42)
    device = get_device()

    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "en_ewt-ud-train.txt"
    )

    print("Loading UD data from local file...")
    sentences = load_ud_file(data_path)

    # dùng 90% train, 10% dev
    split = int(0.9 * len(sentences))
    train_sents = sentences[:split]
    dev_sents = sentences[split:]

    word2id, tag2id = build_vocab(train_sents)

    print(f"Vocab size: {len(word2id)}")
    print(f"Number of POS tags: {len(tag2id) - 1}")

    train_ds = POSDataset(train_sents, word2id, tag2id)
    dev_ds = POSDataset(dev_sents, word2id, tag2id)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=32, collate_fn=collate_fn)

    model = RNNForPOSTagging(
        vocab_size=len(word2id),
        emb_dim=128,
        hidden_dim=128,
        num_tags=len(tag2id)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_LABEL_ID)

    print("Training...")
    for epoch in range(1, 4):
        model.train()
        total_loss = 0

        for xb, lengths, yb in train_loader:
            xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb, lengths)
            loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch} | Loss: {total_loss:.2f} | Dev Token Acc: {acc:.4f}")


if __name__ == "__main__":
    main()

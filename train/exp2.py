# -*- coding: utf-8 -*-
"""第二轮实验：二分类加权 / 多任务双头 / 详细指标（balanced acc、AUC），基于 v1 数据。"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_work" / "names_clean_v1.jsonl"
OUT = ROOT / "train" / "exp_results2.json"

MAX_LEN = 8
LABELS = ["baihe", "purelove", "romance"]
PAD_ID, UNK_ID = 0, 1
SEED = 42


def load_rows():
    rows = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            name = "".join(ch for ch in o["name"] if not ch.isspace())
            if name and o["label"] in LABELS:
                rows.append((name, o["label"]))
    seen, out = set(), []
    for n, l in rows:
        if (n, l) in seen:
            continue
        seen.add((n, l))
        out.append((n, l))
    return out


def build_vocab(names):
    cnt = Counter(ch for n in names for ch in n)
    chars = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = ["<pad>", "<unk>"] + [ch for ch, _ in chars]
    return vocab, {ch: i for i, ch in enumerate(vocab)}


def encode(name, v2i):
    chars = [ch for ch in name if not ch.isspace()][:MAX_LEN]
    ids = [v2i.get(ch, UNK_ID) for ch in chars]
    ids += [PAD_ID] * (MAX_LEN - len(ids))
    return ids


def split_rows(rows, seed=SEED):
    rng = random.Random(seed)
    by = defaultdict(list)
    for n, l in rows:
        by[l].append(n)
    train, val, test = [], [], []
    for l in LABELS:
        arr = by[l][:]
        rng.shuffle(arr)
        n = len(arr)
        nt, nv = max(1, round(n * 0.1)), max(1, round(n * 0.1))
        test += [(x, l) for x in arr[:nt]]
        val += [(x, l) for x in arr[nt:nt + nv]]
        train += [(x, l) for x in arr[nt + nv:]]
    return train, val, test


class MultiTask(nn.Module):
    """共享编码器 + 三分类头 + 二分类头（二分类头给 百合 vs 非百合 用）。"""

    def __init__(self, vocab_size, emb=32, pos=8, chans=(64, 96, 96), hidden=128, drop=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD_ID)
        self.pos = nn.Embedding(MAX_LEN, pos)
        self.convs = nn.ModuleList([nn.Conv1d(emb + pos, c, k) for k, c in zip((1, 2, 3), chans)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(c) for c in chans])
        pooled = 2 * sum(chans) + emb * 2
        self.drop = nn.Dropout(drop)
        self.shared = nn.Linear(pooled, hidden)
        self.head3 = nn.Linear(hidden, 3)
        self.head2 = nn.Linear(hidden, 2)

    def forward(self, x):
        B, L = x.shape
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        e = torch.cat([self.emb(x), self.pos(pos_ids)], dim=2).transpose(1, 2)
        pooled = []
        for conv, bn in zip(self.convs, self.bns):
            z = torch.relu(bn(conv(e)))
            pooled.append(z.max(dim=2).values)
            pooled.append(z.mean(dim=2))
        lens = (x != PAD_ID).sum(dim=1).clamp(min=1)
        first = self.emb(x[:, 0])
        last = self.emb(x[torch.arange(B, device=x.device), lens - 1])
        h = torch.cat(pooled + [first, last], dim=1)
        h = torch.relu(self.shared(self.drop(h)))
        return self.head3(h), self.head2(h)


def evaluate(logits, y, ncls):
    pred = logits.argmax(1)
    acc = (pred == y).float().mean().item()
    f1s, precs, recs = [], [], []
    for c in range(ncls):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * p * r / (p + r) if p + r else 0)
        precs.append(p)
        recs.append(r)
    return round(acc, 4), round(float(np.mean(f1s)), 4), [round(x, 3) for x in f1s], [round(x, 3) for x in precs], [round(x, 3) for x in recs]


def train_multitask(xtr, y3tr, y2tr, xva, y3va, y2va, xtr_w=None, epochs=140, patience=22, lr=1e-3, wd=1e-4, aux=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = MultiTask(V).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    xtr, y3tr, y2tr = xtr.to(device), y3tr.to(device), y2tr.to(device)
    xva, y3va, y2va = xva.to(device), y3va.to(device), y2va.to(device)
    xte_d, y3te_d, y2te_d = xte.to(device), y3te.to(device), y2te.to(device)
    best = (-1, None, 0)
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            o3, o2 = model(xtr[idx])
            loss = lossf(o3, y3tr[idx]) + aux * lossf(o2, y2tr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            o3, o2 = model(xva)
            acc3 = (o3.argmax(1) == y3va).float().mean().item()
            acc2 = (o2.argmax(1) == y2va).float().mean().item()
        score = acc3 + 0.3 * acc2
        if score > best[0] + 1e-6:
            best = (score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        o3, o2 = model(xte_d)
    return model, evaluate(o3, y3te_d, 3), evaluate(o2, y2te_d, 2), best[2]


rows = load_rows()
vocab, v2i = build_vocab([n for n, _ in rows])
label2i = {l: i for i, l in enumerate(LABELS)}
train, val, test = split_rows(rows)
V = len(vocab)


def pack3(pairs):
    x = torch.tensor([encode(n, v2i) for n, _ in pairs], dtype=torch.long)
    y = torch.tensor([label2i[l] for _, l in pairs], dtype=torch.long)
    return x, y


def pack2(pairs):
    x = torch.tensor([encode(n, v2i) for n, _ in pairs], dtype=torch.long)
    y = torch.tensor([1 if l == "baihe" else 0 for _, l in pairs], dtype=torch.long)
    return x, y


xtr, y3tr = pack3(train)
xva, y3va = pack3(val)
xte, y3te = pack3(test)
_, y2tr = pack2(train)
_, y2va = pack2(val)
_, y2te = pack2(test)

model, m3, m2, ep = train_multitask(xtr, y3tr, y2tr, xva, y3va, y2va)
print("multitask 3cls:", m3, "epoch", ep, flush=True)
print("multitask bin:", m2, flush=True)

# 单独二分类（加权 CE，逆频率）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
m = torch.nn.Sequential()  # placeholder to keep flake quiet
num = Counter(y2tr.tolist())
w = torch.tensor([len(y2tr) / (2 * num[0]), len(y2tr) / (2 * num[1])]).to(device)
print("binary class weights:", w.tolist(), flush=True)

bin_model = MultiTask(V).to(device)
opt = torch.optim.Adam(bin_model.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
btr, bytr = xtr.to(device), y2tr.to(device)
bva, byva = xva.to(device), y2va.to(device)
bte, byte = xte.to(device), y2te.to(device)
best = (-1, None, 0)
bad = 0
for ep in range(1, 121):
    bin_model.train()
    perm = torch.randperm(len(btr))
    for i in range(0, len(btr), 128):
        idx = perm[i:i + 128]
        opt.zero_grad()
        _, o2 = bin_model(btr[idx])
        loss = lossf(o2, bytr[idx])
        loss.backward()
        opt.step()
    bin_model.eval()
    with torch.no_grad():
        _, o2 = bin_model(bva)
        acc = (o2.argmax(1) == byva).float().mean().item()
        # balanced accuracy
        pred = o2.argmax(1)
        rec0 = ((pred == 0) & (byva == 0)).sum().item() / max(1, (byva == 0).sum().item())
        rec1 = ((pred == 1) & (byva == 1)).sum().item() / max(1, (byva == 1).sum().item())
        bal = (rec0 + rec1) / 2
    if bal > best[0]:
        best = (bal, {k: v.detach().cpu().clone() for k, v in bin_model.state_dict().items()}, ep)
        bad = 0
    else:
        bad += 1
    if bad >= 25:
        break
bin_model.load_state_dict(best[1])
bin_model.eval()
with torch.no_grad():
    _, o2 = bin_model(bte)
print("weighted binary test:", evaluate(o2, byte, 2), "epoch", best[2], flush=True)

res = {"multitask_3cls": m3, "multitask_bin": m2, "weighted_bin": evaluate(o2, byte, 2)}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved", OUT, flush=True)

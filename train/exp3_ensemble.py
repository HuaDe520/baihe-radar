# -*- coding: utf-8 -*-
"""v2 模型 3 seed 集成实验（v1 数据）：看集成对 3 类 acc 与二分 balanced acc 的提升。"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import train_v2 as T  # noqa: E402


def train_one(seed, xtr, y3tr, y2tr, xva, y3va, y2va, V, epochs=140, patience=22):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T.DualHead(V, drop=0.4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-4)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    def d(t):
        return t.to(device)
    xtr, y3tr, y2tr = d(xtr), d(y3tr), d(y2tr)
    xva, y3va, y2va = d(xva), d(y3va), d(y2va)
    best, bad, best_state = -1.0, 0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            o3, o2 = model(xtr[idx])
            loss = lossf(o3, y3tr[idx]) + 0.5 * lossf(o2, y2tr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            o3, o2 = model(xva)
            acc3 = (o3.argmax(1) == y3va).float().mean().item()
            p2 = o2.argmax(1)
            rec0 = ((p2 == 0) & (y2va == 0)).sum().item() / max(1, (y2va == 0).sum().item())
            rec1 = ((p2 == 1) & (y2va == 1)).sum().item() / max(1, (y2va == 1).sum().item())
            bal2 = (rec0 + rec1) / 2
        score = acc3 + 0.5 * bal2
        if score > best + 1e-6:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model


def main():
    rows = T.load_rows(ROOT / "_work" / "names_clean_v1.jsonl")
    vocab, v2i = T.build_vocab([n for n, _ in rows])
    V = len(vocab)
    label2i = {l: i for i, l in enumerate(T.LABELS)}
    tr, va, te = T.split_rows(rows)

    def pack(pairs):
        x = torch.tensor([T.encode(n, v2i) for n, _ in pairs], dtype=torch.long)
        y3 = torch.tensor([label2i[l] for _, l in pairs], dtype=torch.long)
        y2 = torch.tensor([1 if l == "baihe" else 0 for _, l in pairs], dtype=torch.long)
        return x, y3, y2

    xtr, y3tr, y2tr = pack(tr)
    xva, y3va, y2va = pack(va)
    xte, y3te, y2te = pack(te)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs3, probs2 = [], []
    for seed in (1, 2, 3):
        m = train_one(seed, xtr, y3tr, y2tr, xva, y3va, y2va, V)
        with torch.no_grad():
            o3, o2 = m(xte.to(device))
        probs3.append(torch.softmax(o3, 1).cpu().numpy())
        probs2.append(torch.softmax(o2, 1).cpu().numpy())
        i3 = probs3[-1].argmax(1)
        i2 = probs2[-1].argmax(1)
        acc3 = (i3 == y3te.numpy()).mean()
        rec0 = ((i2 == 0) & (y2te.numpy() == 0)).sum() / max(1, (y2te.numpy() == 0).sum())
        rec1 = ((i2 == 1) & (y2te.numpy() == 1)).sum() / max(1, (y2te.numpy() == 1).sum())
        print(f"seed {seed}: acc3={acc3:.4f} bal2={(rec0+rec1)/2:.4f}", flush=True)

    p3 = np.mean(probs3, 0)
    p2 = np.mean(probs2, 0)
    i3 = p3.argmax(1)
    i2 = p2.argmax(1)
    y3 = y3te.numpy()
    y2 = y2te.numpy()
    acc3 = (i3 == y3).mean()
    rec0 = ((i2 == 0) & (y2 == 0)).sum() / max(1, (y2 == 0).sum())
    rec1 = ((i2 == 1) & (y2 == 1)).sum() / max(1, (y2 == 1).sum())
    print(f"ENSEMBLE(3): acc3={acc3:.4f} bal2={(rec0+rec1)/2:.4f}", flush=True)


if __name__ == "__main__":
    main()

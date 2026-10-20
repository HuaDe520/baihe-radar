# -*- coding: utf-8 -*-
"""架构对比实验：在同一划分上比较 CNN / CNN+位置+首末字 / BN 深 CNN / BiGRU / 二分类。

输出：train/exp_results.json，并把每个配置的验证/测试指标 print 出来。
不写 web/ 下任何文件。
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_work" / "names_clean_v1.jsonl"
OUT = ROOT / "train" / "exp_results.json"

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


class CnnPosFeat(nn.Module):
    """emb + 位置嵌入 + 三路卷积(max/mean) + 首字/末字嵌入拼接 → FC。"""

    def __init__(self, vocab_size, emb=32, pos=8, chans=(48, 64, 64), hidden=128, ncls=3, drop=0.35):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD_ID)
        self.pos = nn.Embedding(MAX_LEN, pos)
        self.convs = nn.ModuleList([nn.Conv1d(emb + pos, c, k) for k, c in zip((1, 2, 3), chans)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(c) for c in chans])
        pooled = 2 * sum(chans) + emb * 2
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(pooled, hidden)
        self.out = nn.Linear(hidden, ncls)

    def forward(self, x):
        B, L = x.shape
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        e = torch.cat([self.emb(x), self.pos(pos_ids)], dim=2).transpose(1, 2)  # B,E,L
        pooled = []
        for conv, bn in zip(self.convs, self.bns):
            z = torch.relu(bn(conv(e)))
            pooled.append(z.max(dim=2).values)
            pooled.append(z.mean(dim=2))
        # 首字 / 最后一个非 pad 字
        first = self.emb(x[:, 0])
        lens = (x != PAD_ID).sum(dim=1).clamp(min=1)
        last = self.emb(x[torch.arange(B, device=x.device), lens - 1])
        h = torch.cat(pooled + [first, last], dim=1)
        h = torch.relu(self.fc(self.drop(h)))
        return self.out(h)


class CnnBN(nn.Module):
    """对照：更宽的 CNN + BN + dropout，无位置/首末字特征。"""

    def __init__(self, vocab_size, emb=32, chans=(64, 96, 96), hidden=128, ncls=3, drop=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD_ID)
        self.convs = nn.ModuleList([nn.Conv1d(emb, c, k) for k, c in zip((1, 2, 3), chans)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(c) for c in chans])
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * sum(chans), hidden)
        self.out = nn.Linear(hidden, ncls)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        pooled = []
        for conv, bn in zip(self.convs, self.bns):
            z = torch.relu(bn(conv(e)))
            pooled.append(z.max(dim=2).values)
            pooled.append(z.mean(dim=2))
        h = torch.cat(pooled, dim=1)
        h = torch.relu(self.fc(self.drop(h)))
        return self.out(h)


class BiGRU(nn.Module):
    def __init__(self, vocab_size, emb=48, hidden=64, ncls=3, drop=0.35):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD_ID)
        self.rnn = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(hidden * 4 + emb * 2, 128)
        self.out = nn.Linear(128, ncls)

    def forward(self, x):
        e = self.emb(x)
        z, _ = self.rnn(e)
        mask = (x != PAD_ID).unsqueeze(2).float()
        zmax = (z * mask + (mask - 1) * 1e9).max(dim=1).values
        zmean = (z * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        lens = (x != PAD_ID).sum(dim=1).clamp(min=1)
        last = e[torch.arange(x.size(0), device=x.device), lens - 1]
        h = torch.cat([zmax, zmean, self.emb(x[:, 0]), last], dim=1)
        h = torch.relu(self.fc(self.drop(h)))
        return self.out(h)


def run_one(name, model, xtr, ytr, xva, yva, xte, yte, ncls=3, lr=1e-3, wd=1e-4, epochs=120, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    xtr, ytr, xva, yva, xte, yte = [t.to(device) for t in (xtr, ytr, xva, yva, xte, yte)]
    best = (-1.0, 1e9, None, 0)
    bad = 0
    bs = 128
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(xtr[idx])
            loss = lossf(out, ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vo = model(xva)
            vacc = (vo.argmax(1) == yva).float().mean().item()
            vloss = lossf(vo, yva).item()
            te = model(xte)
            tacc = (te.argmax(1) == yte).float().mean().item()
        score = vacc
        if score > best[0] + 1e-6:
            best = (vacc, tacc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best[2])
    model.eval()
    with torch.no_grad():
        pred = model(xte).argmax(1)
        acc = (pred == yte).float().mean().item()
        # macro F1
        k = ncls
        f1s = []
        for c in range(k):
            tp = ((pred == c) & (yte == c)).sum().item()
            fp = ((pred == c) & (yte != c)).sum().item()
            fn = ((pred != c) & (yte == c)).sum().item()
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
        macro_f1 = float(np.mean(f1s))
    res = {"arch": name, "val_acc": round(best[0], 4), "test_acc": round(acc, 4),
           "macro_f1": round(macro_f1, 4), "best_epoch": best[3],
           "params": sum(p.numel() for p in model.parameters())}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


def main():
    rows = load_rows()
    vocab, v2i = build_vocab([n for n, _ in rows])
    label2i = {l: i for i, l in enumerate(LABELS)}
    train, val, test = split_rows(rows)

    def pack(pairs):
        x = torch.tensor([encode(n, v2i) for n, _ in pairs], dtype=torch.long)
        y = torch.tensor([label2i[l] for _, l in pairs], dtype=torch.long)
        return x, y

    xtr, ytr = pack(train)
    xva, yva = pack(val)
    xte, yte = pack(test)
    print(f"data={len(rows)} vocab={len(vocab)} train={len(train)} val={len(val)} test={len(test)}", flush=True)

    torch.manual_seed(SEED)
    results = []
    results.append(run_one("cnn_pos_feat", CnnPosFeat(len(vocab)), xtr, ytr, xva, yva, xte, yte))
    torch.manual_seed(SEED)
    results.append(run_one("cnn_bn_wide", CnnBN(len(vocab)), xtr, ytr, xva, yva, xte, yte))
    torch.manual_seed(SEED)
    results.append(run_one("bigru", BiGRU(len(vocab)), xtr, ytr, xva, yva, xte, yte))

    # 二分类：百合 vs 非百合（纯爱+言情合并）
    bin_train = [(n, 1 if l == "baihe" else 0) for n, l in train]
    bin_val = [(n, 1 if l == "baihe" else 0) for n, l in val]
    bin_test = [(n, 1 if l == "baihe" else 0) for n, l in test]

    def pack2(pairs):
        x = torch.tensor([encode(n, v2i) for n, _ in pairs], dtype=torch.long)
        y = torch.tensor([l for _, l in pairs], dtype=torch.long)
        return x, y

    btr, bytr = pack2(bin_train)
    bva, byva = pack2(bin_val)
    bte, byte = pack2(bin_test)
    torch.manual_seed(SEED)
    results.append(run_one("binary_cnn_pos_feat", CnnPosFeat(len(vocab), ncls=2), btr, bytr, bva, byva, bte, byte, ncls=2))

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()

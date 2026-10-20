# -*- coding: utf-8 -*-
"""训练字符级 CNN 三分类（百合 / 纯爱 / 言情），并导出 web/model.json。

用法： python train/train.py
输入： data/names_clean.jsonl（{"name","label",...}，label ∈ baihe/purelove/romance）
输出： web/model.json、train/metrics.json、train/splits.json、train/train_report.md
"""
import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "names_clean.jsonl"
MODEL_OUT = ROOT / "web" / "model.json"
TRAIN_DIR = ROOT / "train"

MAX_LEN = 8
EMB_DIM = 28
HIDDEN = 64
CONVS = [(1, 32), (2, 40), (3, 40)]
LABELS = ["baihe", "purelove", "romance"]
CLASS_NAMES = ["百合", "纯爱(BL)", "言情(BG)"]
PAD_ID, UNK_ID = 0, 1
SEED = 42


def load_rows():
    rows = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            name = "".join(ch for ch in o["name"] if not ch.isspace())
            if not name or o["label"] not in LABELS:
                continue
            rows.append((name, o["label"]))
    # 同名同标签去重
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


class CharCNN(nn.Module):
    def __init__(self, vocab_size, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=PAD_ID)
        self.convs = nn.ModuleList([nn.Conv1d(EMB_DIM, c, k) for k, c in CONVS])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * sum(c for _, c in CONVS), HIDDEN)
        self.out = nn.Linear(HIDDEN, len(LABELS))

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)  # B,E,L
        pooled = []
        for conv in self.convs:
            z = conv(e)                                  # B,C,L-k+1
            z = torch.relu(z)
            pooled.append(z.max(dim=2).values)           # max-pool
            pooled.append(z.mean(dim=2))                 # mean-pool
        h = torch.cat(pooled, dim=1)
        h = torch.relu(self.fc(self.drop(h)))
        return self.out(h)


def stratified_split(rows, seed=SEED):
    rng = random.Random(seed)
    by = defaultdict(list)
    for n, l in rows:
        by[l].append(n)
    train, val, test = [], [], []
    for l in LABELS:
        arr = by[l][:]
        rng.shuffle(arr)
        n = len(arr)
        n_test = max(1, int(round(n * 0.1)))
        n_val = max(1, int(round(n * 0.1)))
        test += [(x, l) for x in arr[:n_test]]
        val += [(x, l) for x in arr[n_test:n_test + n_val]]
        train += [(x, l) for x in arr[n_test + n_val:]]
    return train, val, test


def metrics_of(y_true, y_pred):
    n = len(y_true)
    k = len(LABELS)
    cm = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    per_class = {}
    f1s = []
    for i, lab in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(k)) - tp
        fn = sum(cm[i]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        per_class[lab] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    acc = sum(cm[i][i] for i in range(k)) / n if n else 0.0
    return acc, sum(f1s) / k, cm, per_class


def rounded(arr):
    return [round(float(x), 6) for x in np.asarray(arr).reshape(-1)]


def export(model, vocab, metrics):
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    emb = sd["emb.weight"]
    convs = []
    for i, (k, c) in enumerate(CONVS):
        w = sd[f"convs.{i}.weight"]                       # C,E,k
        w = np.transpose(w, (0, 2, 1)).reshape(c, k * EMB_DIM)  # C, k*E 行主序 j*E+e
        convs.append({"k": k,
                      "W": {"shape": [c, k * EMB_DIM], "data": rounded(w)},
                      "b": {"shape": [c], "data": rounded(sd[f"convs.{i}.bias"])}})
    out = {
        "format": "baihe-charcnn-v1",
        "config": {
            "max_len": MAX_LEN, "emb_dim": EMB_DIM, "hidden": HIDDEN,
            "num_classes": len(LABELS), "pad_id": PAD_ID, "unk_id": UNK_ID,
            "conv": [{"k": k, "ch": c} for k, c in CONVS],
            "classes": LABELS, "class_names_zh": CLASS_NAMES,
        },
        "vocab": vocab,
        "weights": {
            "emb": {"shape": list(emb.shape), "data": rounded(emb)},
            "conv": convs,
            "fc": {"W": {"shape": list(sd["fc.weight"].shape), "data": rounded(sd["fc.weight"])},
                   "b": {"shape": list(sd["fc.bias"].shape), "data": rounded(sd["fc.bias"])}},
            "out": {"W": {"shape": list(sd["out.weight"].shape), "data": rounded(sd["out.weight"])},
                    "b": {"shape": list(sd["out.bias"].shape), "data": rounded(sd["out.bias"])}},
        },
        "metrics": metrics,
        "dataset": metrics.pop("_dataset", {}),
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rows = load_rows()
    if len(rows) < 300:
        sys.exit(f"数据太少：{len(rows)} 条，请先运行 data/scrape.py")
    counts = Counter(l for _, l in rows)
    print("数据：", dict(counts), "总", len(rows))

    vocab, v2i = build_vocab([n for n, _ in rows])
    label2i = {l: i for i, l in enumerate(LABELS)}
    print("词表大小：", len(vocab))

    train, val, test = stratified_split(rows, seed=args.seed)
    print(f"划分：train={len(train)} val={len(val)} test={len(test)}")

    def pack(pairs):
        x = torch.tensor([encode(n, v2i) for n, _ in pairs], dtype=torch.long)
        y = torch.tensor([label2i[l] for _, l in pairs], dtype=torch.long)
        return x, y

    xtr, ytr = pack(train)
    xva, yva = pack(val)
    xte, yte = pack(test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备：", device)
    model = CharCNN(len(vocab), dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    xva_d, yva_d = xva.to(device), yva.to(device)

    best_val_acc, best_val_loss, best_state, best_epoch = -1.0, 1e9, None, 0
    patience, bad = args.patience, 0
    bs = 128
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        tot, cnt = 0.0, 0
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            cnt += len(idx)
        model.eval()
        with torch.no_grad():
            vo = model(xva_d)
            va = lossf(vo, yva_d).item()
            vacc = (vo.argmax(dim=1) == yva_d).float().mean().item()
        history.append({"epoch": epoch, "train_loss": round(tot / cnt, 4),
                        "val_loss": round(va, 4), "val_acc": round(vacc, 4)})
        improved = vacc > best_val_acc + 1e-6 or (abs(vacc - best_val_acc) <= 1e-6 and va < best_val_loss)
        if improved:
            best_val_acc, best_val_loss, best_epoch, bad = vacc, va, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d} train_loss={tot / cnt:.4f} val_loss={va:.4f} val_acc={vacc:.4f} best_acc={best_val_acc:.4f}")
        if bad >= patience:
            print(f"早停于 epoch {epoch}（best epoch {best_epoch}, val_acc {best_val_acc:.4f}）")
            break

    model.load_state_dict(best_state)
    print(f"最佳验证准确率 {best_val_acc:.4f} @ epoch {best_epoch}")
    model.eval()
    with torch.no_grad():
        pred = model(xte.to(device)).argmax(dim=1).cpu().tolist()
    y_true = [label2i[l] for _, l in test]
    acc, macro_f1, cm, per_class = metrics_of(y_true, pred)
    baseline = max(counts.values()) / len(rows)
    print(f"\n测试集准确率 {acc:.4f} | macro-F1 {macro_f1:.4f} | 多数类基线 {baseline:.4f}")
    print("混淆矩阵(行=真实,列=预测,顺序", LABELS, "):")
    for r in cm:
        print("  ", r)

    metrics = {
        "test_acc": round(acc, 4),
        "baseline": round(baseline, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion": cm,
        "n_train": len(train),
        "n_test": len(test),
        "n_val": len(val),
        "_dataset": {"baihe": counts["baihe"], "purelove": counts["purelove"],
                     "romance": counts["romance"],
                     "source": "晋江文学城(jjwxc.net)作品库 主角名"},
    }
    out = export(model, vocab, metrics)

    # 保存 splits 和 metrics
    TRAIN_DIR.mkdir(exist_ok=True)
    with open(TRAIN_DIR / "splits.json", "w", encoding="utf-8") as f:
        json.dump({"train": train, "val": val, "test": test}, f, ensure_ascii=False)
    with open(TRAIN_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in metrics.items() if k != "_dataset"}, f, ensure_ascii=False, indent=2)

    report = f"""# 训练报告

- 数据：{len(rows)} 个主角名（{dict(counts)}），来自晋江文学城作品库主角名字段
- 词表：{len(vocab)}（pad + unk + {len(vocab) - 2} 个汉字）
- 划分：train {len(train)} / val {len(val)} / test {len(test)}
- 模型：字符 embedding {EMB_DIM} + conv k=1/2/3 (32/40/40) + ReLU + max/mean 双池化 + FC{HIDDEN} + 输出 3
- 参数量：约 {sum(p.numel() for p in model.parameters()):,}
- 测试集准确率：**{acc:.2%}**（多数类基线 {baseline:.2%}，随机基线 33.3%）
- macro-F1：{macro_f1:.4f}
- 各类：{json.dumps(per_class, ensure_ascii=False)}
- 混淆矩阵（行=真实，列=预测，顺序 百合/纯爱/言情）：{cm}
- 导出：web/model.json（{MODEL_OUT.stat().st_size / 1024:.0f} KB）

> 说明：这是娱乐向玩具，模型学的是网络小说频道的取名风格，不构成对真实个人性取向的任何判断。
"""
    (TRAIN_DIR / "train_report.md").write_text(report, encoding="utf-8")
    print("\n导出完成：", MODEL_OUT, f"{MODEL_OUT.stat().st_size / 1024:.0f} KB")
    print(report)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""v2 双头模型：共享编码器（emb+pos → conv k=1/2/3 + BN → max/mean pool → 首末字 emb）
+ 三分类头（百合/纯爱/言情）+ 二分类头（百合 vs 非百合）。

用法：
  python train/train_v2.py --data data/names_clean.jsonl --out _work/model_v2.json
输出：模型 JSON、train/metrics_v2.json、train/splits_v2.json、train/train_report_v2.md
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["baihe", "purelove", "romance"]
CLASS_NAMES = ["百合", "纯爱(BL)", "言情(BG)"]
MAX_LEN = 8
EMB_DIM = 32
POS_DIM = 8
CONVS = [(1, 64), (2, 96), (3, 96)]
HIDDEN = 128
PAD_ID, UNK_ID = 0, 1
SEED = 42
BN_EPS = 1e-5


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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


class DualHead(nn.Module):
    def __init__(self, vocab_size, drop=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=PAD_ID)
        self.pos = nn.Embedding(MAX_LEN, POS_DIM)
        self.convs = nn.ModuleList([nn.Conv1d(EMB_DIM + POS_DIM, c, k) for k, c in CONVS])
        self.bns = nn.ModuleList([nn.BatchNorm1d(c) for _, c in CONVS])
        pooled = 2 * sum(c for _, c in CONVS) + EMB_DIM * 2
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(pooled, HIDDEN)
        self.out3 = nn.Linear(HIDDEN, len(LABELS))
        self.out2 = nn.Linear(HIDDEN, 2)

    def forward(self, x):
        B, L = x.shape
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        e = torch.cat([self.emb(x), self.pos(pos_ids)], dim=2).transpose(1, 2)  # B,40,8
        pooled = []
        for conv, bn in zip(self.convs, self.bns):
            z = torch.relu(bn(conv(e)))
            pooled.append(z.max(dim=2).values)
            pooled.append(z.mean(dim=2))
        lens = (x != PAD_ID).sum(dim=1).clamp(min=1)
        first = self.emb(x[:, 0])
        last = self.emb(x[torch.arange(B, device=x.device), lens - 1])
        h0 = torch.cat(pooled + [first, last], dim=1)
        h = torch.relu(self.fc(self.drop(h0)))
        return self.out3(h), self.out2(h)


def metrics_3(y_true, y_pred):
    n = len(y_true)
    cm = [[0] * 3 for _ in range(3)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    per, f1s = {}, []
    for i, lab in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(3)) - tp
        fn = sum(cm[i]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        per[lab] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    acc = sum(cm[i][i] for i in range(3)) / n if n else 0.0
    return round(acc, 4), round(float(np.mean(f1s)), 4), cm, per


def metrics_2(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    rec1 = tp / (tp + fn) if tp + fn else 0.0
    rec0 = tn / (tn + fp) if tn + fp else 0.0
    prec1 = tp / (tp + fp) if tp + fp else 0.0
    prec0 = tn / (tn + fn) if tn + fn else 0.0
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0
    macro_f1 = (f1(prec0, rec0) + f1(prec1, rec1)) / 2
    return round(acc, 4), round((rec0 + rec1) / 2, 4), round(macro_f1, 4), [[tn, fp], [fn, tp]]


def rounded(a):
    return [round(float(x), 6) for x in np.asarray(a).reshape(-1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "names_clean.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "_work" / "model_v2.json"))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=2e-4)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--aux", type=float, default=0.5, help="二分类辅助损失权重")
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rows = load_rows(args.data)
    counts = Counter(l for _, l in rows)
    vocab, v2i = build_vocab([n for n, _ in rows])
    V = len(vocab)
    print(f"数据：{dict(counts)} 总 {len(rows)}；词表 {V}", flush=True)

    train, val, test = split_rows(rows, args.seed)
    label2i = {l: i for i, l in enumerate(LABELS)}

    def pack(pairs):
        x = torch.tensor([encode(n, v2i) for n, _ in pairs], dtype=torch.long)
        y3 = torch.tensor([label2i[l] for _, l in pairs], dtype=torch.long)
        y2 = torch.tensor([1 if l == "baihe" else 0 for _, l in pairs], dtype=torch.long)
        return x, y3, y2

    xtr, y3tr, y2tr = pack(train)
    xva, y3va, y2va = pack(val)
    xte, y3te, y2te = pack(test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备：", device, flush=True)
    model = DualHead(V, drop=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    def to_dev(x, y3, y2):
        return x.to(device), y3.to(device), y2.to(device)

    xtr, y3tr, y2tr = to_dev(xtr, y3tr, y2tr)
    xva, y3va, y2va = to_dev(xva, y3va, y2va)
    xte, y3te, y2te = to_dev(xte, y3te, y2te)

    best_score, best_state, best_ep, bad = -1.0, None, 0, 0
    best_val = {"acc3": 0.0, "bal2": 0.0}
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            o3, o2 = model(xtr[idx])
            loss = lossf(o3, y3tr[idx]) + args.aux * lossf(o2, y2tr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            o3, o2 = model(xva)
            acc3 = (o3.argmax(1) == y3va).float().mean().item()
            b2 = metrics_2((o2.argmax(1) == y2va).long().tolist(), (o2.argmax(1) == y2va).long().tolist())[1]
            # 真实 balanced acc
        with torch.no_grad():
            p2 = o2.argmax(1)
            rec0 = ((p2 == 0) & (y2va == 0)).sum().item() / max(1, (y2va == 0).sum().item())
            rec1 = ((p2 == 1) & (y2va == 1)).sum().item() / max(1, (y2va == 1).sum().item())
            bal2 = (rec0 + rec1) / 2
        score = acc3 + 0.5 * bal2
        history.append({"epoch": ep, "val_acc3": round(acc3, 4), "val_bal2": round(bal2, 4)})
        if score > best_score + 1e-6:
            best_score, best_ep, bad = score, ep, 0
            best_val = {"acc3": round(acc3, 4), "bal2": round(bal2, 4)}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0:
            print(f"ep {ep:3d} val_acc3={acc3:.4f} val_bal2={bal2:.4f} best={best_score:.4f}", flush=True)
        if bad >= args.patience:
            print(f"早停 ep {ep}（best epoch {best_ep}）", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        o3, o2 = model(xte)
    p3 = o3.argmax(1).cpu().tolist()
    p2 = o2.argmax(1).cpu().tolist()
    acc3, mf3, cm3, per = metrics_3(y3te.cpu().tolist(), p3)
    bacc, bbal, bmf, bcm = metrics_2(y2te.cpu().tolist(), p2)
    baseline3 = round(max(counts.values()) / len(rows), 4)
    print(f"3类: acc={acc3} macroF1={mf3} baseline={baseline3}", flush=True)
    print(f"二分类: acc={bacc} balanced={bbal} macroF1={bmf} cm={bcm}", flush=True)

    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    convs = []
    in_ch = EMB_DIM + POS_DIM
    for i, (k, c) in enumerate(CONVS):
        w = sd[f"convs.{i}.weight"]                     # C, in_ch, k
        w2 = np.transpose(w, (0, 2, 1)).reshape(c, k * in_ch)  # C, k*in_ch
        convs.append({
            "k": k,
            "W": {"shape": [c, k * in_ch], "data": rounded(w2)},
            "b": {"shape": [c], "data": rounded(sd[f"convs.{i}.bias"])},
            "bn": {
                "gamma": {"shape": [c], "data": rounded(sd[f"bns.{i}.weight"])},
                "beta": {"shape": [c], "data": rounded(sd[f"bns.{i}.bias"])},
                "mean": {"shape": [c], "data": rounded(sd[f"bns.{i}.running_mean"])},
                "var": {"shape": [c], "data": rounded(sd[f"bns.{i}.running_var"])},
            },
        })
    metrics = {
        "test_acc_3cls": acc3,
        "baseline_3cls": baseline3,
        "macro_f1_3cls": mf3,
        "binary_acc": bacc,
        "binary_balanced_acc": bbal,
        "binary_macro_f1": bmf,
        "per_class": per,
        "confusion": cm3,
        "binary_confusion": bcm,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "params": int(sum(p.numel() for p in model.parameters())),
        "best_epoch": best_ep,
        "val_acc3": best_val["acc3"],
        "val_bal2": best_val["bal2"],
    }
    out = {
        "format": "baihe-dualhead-v2",
        "config": {
            "max_len": MAX_LEN, "emb_dim": EMB_DIM, "pos_dim": POS_DIM,
            "conv": [{"k": k, "ch": c} for k, c in CONVS],
            "hidden": HIDDEN, "num_classes": len(LABELS), "num_binary": 2,
            "pad_id": PAD_ID, "unk_id": UNK_ID, "bn_eps": BN_EPS,
            "classes": LABELS, "class_names_zh": CLASS_NAMES,
            "binary_labels": ["not_baihe", "baihe"],
            "feat": ["first_last_emb"],
        },
        "vocab": vocab,
        "weights": {
            "emb": {"shape": [V, EMB_DIM], "data": rounded(sd["emb.weight"])},
            "pos": {"shape": [MAX_LEN, POS_DIM], "data": rounded(sd["pos.weight"])},
            "conv": convs,
            "fc": {"W": {"shape": [HIDDEN, 2 * sum(c for _, c in CONVS) + 2 * EMB_DIM],
                         "data": rounded(sd["fc.weight"])},
                   "b": {"shape": [HIDDEN], "data": rounded(sd["fc.bias"])}},
            "out3": {"W": {"shape": [len(LABELS), HIDDEN], "data": rounded(sd["out3.weight"])},
                     "b": {"shape": [len(LABELS)], "data": rounded(sd["out3.bias"])}},
            "out2": {"W": {"shape": [2, HIDDEN], "data": rounded(sd["out2.weight"])},
                     "b": {"shape": [2], "data": rounded(sd["out2.bias"])}},
        },
        "metrics": metrics,
        "dataset": {"baihe": counts["baihe"], "purelove": counts["purelove"], "romance": counts["romance"],
                    "source": "晋江文学城(jjwxc.net)作品库 主角名"},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"导出 {out_path} {out_path.stat().st_size/1024:.0f} KB", flush=True)

    train_dir = ROOT / "train"
    (train_dir / "metrics_v2.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (train_dir / "splits_v2.json").write_text(
        json.dumps({"train": train, "val": val, "test": test}, ensure_ascii=False), encoding="utf-8")
    report = f"""# v2 双头模型训练报告

- 数据：{len(rows)} 个主角名（百合 {counts['baihe']} / 纯爱 {counts['purelove']} / 言情 {counts['romance']}）
- 词表：{V}；划分 train {len(train)} / val {len(val)} / test {len(test)}
- 结构：emb {EMB_DIM} + pos {POS_DIM} → conv k=1/2/3 (64/96/96) + BN → ReLU → max/mean pool
  → 拼首字/末字 emb（576 维）→ FC {HIDDEN} → 双头（3 分类 + 2 分类）
- 参数：{metrics['params']:,}
- 三分类：测试 acc **{acc3:.2%}**（多数类基线 {baseline3:.2%}，随机 33.3%），macro-F1 {mf3:.4f}
- 二分类（百合 vs 非百合）：acc {bacc:.2%}，**平衡准确率 {bbal:.2%}**，macro-F1 {bmf:.4f}，混淆 [[tn,fp],[fn,tp]]={bcm}
- 混淆矩阵（3 类，行=真实）：{cm3}
- 最佳 epoch：{best_ep}（验证 acc3 {best_val['acc3']:.2%} / bal2 {best_val['bal2']:.2%}）

> 说明：名字与频道取向只有弱相关，3 分类接近「性别用字统计」；二分类平衡准确率更能反映真实可分性。
> 纯娱乐向玩具，不构成对真实个人性取向的判断。
"""
    (train_dir / "train_report_v2.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""v4 五头模型：性别 + ACG 性别表达 + ACG 百合角色 + 作品频道 + 百合判定；支持网名增广。

用法：python train/train_v4.py --out web/model.json
输出：模型 JSON、train/metrics_v4.json、train/splits_v4.json、train/train_report_v4.md
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path(r"D:\names-corpus")

MAX_LEN = 12
EMB_DIM = 32
POS_DIM = 8
CONVS = [(1, 64), (2, 96), (3, 96)]
HIDDEN = 160
PAD_ID, UNK_ID = 0, 1
BN_EPS = 1e-5
SEED = 42
SUFFIXES = ["_233", "233", "_01", "666", "77", "x", "_"]

HEADS = {
    "gender": {"classes": ["female", "male"], "names_zh": ["女", "男"]},
    "expression": {"classes": ["trap", "futanari", "tomboy", "normal"],
                   "names_zh": ["伪娘", "扶她", "伪郎", "普通"]},
    "yuri_char": {"classes": ["other_char", "yuri_char"], "names_zh": ["其他角色", "百合角色"]},
    "genre": {"classes": ["baihe", "purelove", "romance"],
              "names_zh": ["百合", "纯爱(BL)", "言情(BG)"]},
    "yuri": {"classes": ["not_baihe", "baihe"], "names_zh": ["非百合", "百合"]},
}


def normalize_name(raw):
    s = "".join(ch for ch in raw.strip() if not ch.isspace())
    return "".join(ch.lower() if "A" <= ch <= "Z" else ch for ch in s)


def valid_name(name):
    if not (2 <= len(name) <= MAX_LEN):
        return False
    return bool(re.fullmatch(r"[\u4e00-\u9fa5a-z0-9_\-\.·]+", name))


def load_jsonl_rows(path, label_key="label"):
    rows = []
    p = Path(path)
    if not p.exists():
        print("missing", p, flush=True)
        return rows
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            name = normalize_name(o["name"])
            if valid_name(name):
                rows.append((name, o[label_key]))
    return rows


def load_gender_zh(limit_per_class=20000):
    path = CORPUS / "Chinese_Names_Corpus" / "Chinese_Names_Corpus_Gender（120W）.txt"
    male, female = [], []
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            name, sex = normalize_name(parts[0]), parts[1].strip()
            if not re.fullmatch(r"[\u4e00-\u9fa5·]{2,6}", name):
                continue
            if sex == "男":
                male.append(name)
            elif sex == "女":
                female.append(name)
    rng = random.Random(SEED)
    rng.shuffle(male)
    rng.shuffle(female)
    rows = [(n, "male") for n in male[:limit_per_class]] + [(n, "female") for n in female[:limit_per_class]]
    rng.shuffle(rows)
    print("gender_zh", len(rows), flush=True)
    return rows


def load_gender_foreign(limit_per_class=20000):
    path = CORPUS / "English_Names_Corpus" / "English_Cn_Name_Corpus_Gender（48W）.txt"
    male, female = [], []
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            cn, en, sex = parts
            label = {"M": "male", "F": "female"}.get(sex.strip().upper())
            if not label:
                continue
            for v in (normalize_name(cn), re.sub(r"[^a-z]", "", en.lower())):
                if re.fullmatch(r"[\u4e00-\u9fa5·]{2,10}", v) or re.fullmatch(r"[a-z]{2,12}", v):
                    (male if label == "male" else female).append(v)
    rng = random.Random(SEED + 1)
    rng.shuffle(male)
    rng.shuffle(female)
    rows = [(n, "male") for n in male[:limit_per_class]] + [(n, "female") for n in female[:limit_per_class]]
    rng.shuffle(rows)
    print("gender_foreign", len(rows), flush=True)
    return rows


def dedupe(rows):
    seen, out = set(), []
    for n, l in rows:
        if (n, l) in seen:
            continue
        seen.add((n, l))
        out.append((n, l))
    return out


def augment(rows, seed=SEED + 7, p=0.3):
    """网名增广：给一部分样本加数字/下划线后缀，让模型对网名鲁棒。"""
    rng = random.Random(seed)
    out = list(rows)
    for n, l in rows:
        if rng.random() < p:
            out.append((n + rng.choice(SUFFIXES), l))
        if rng.random() < 0.08:
            out.append(("x" + n + rng.choice(SUFFIXES), l))
    return out


def build_vocab(names):
    cnt = Counter(ch for n in names for ch in n)
    chars = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = ["<pad>", "<unk>"] + [ch for ch, _ in chars]
    return vocab, {ch: i for i, ch in enumerate(vocab)}


def encode(name, v2i):
    chars = list(name)[:MAX_LEN]
    ids = [v2i.get(ch, UNK_ID) for ch in chars]
    ids += [PAD_ID] * (MAX_LEN - len(ids))
    return ids


class MultiHead(nn.Module):
    def __init__(self, vocab_size, drop=0.35):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=PAD_ID)
        self.pos = nn.Embedding(MAX_LEN, POS_DIM)
        self.convs = nn.ModuleList([nn.Conv1d(EMB_DIM + POS_DIM, c, k) for k, c in CONVS])
        self.bns = nn.ModuleList([nn.BatchNorm1d(c) for _, c in CONVS])
        pooled = 2 * sum(c for _, c in CONVS) + EMB_DIM * 2
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(pooled, HIDDEN)
        self.heads = nn.ModuleDict({k: nn.Linear(HIDDEN, len(v["classes"])) for k, v in HEADS.items()})

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
        idx = torch.arange(B, device=x.device)
        h = torch.cat(pooled + [self.emb(x[:, 0]), self.emb(x[idx, lens - 1])], dim=1)
        h = torch.relu(self.fc(self.drop(h)))
        return {k: head(h) for k, head in self.heads.items()}


def split(rows, seed=SEED):
    rng = random.Random(seed)
    by = defaultdict(list)
    for n, l in rows:
        by[l].append((n, l))
    tr, va, te = [], [], []
    for l, arr in by.items():
        arr = arr[:]
        rng.shuffle(arr)
        n = len(arr)
        nt, nv = max(1, int(n * 0.1)), max(1, int(n * 0.1))
        te += arr[:nt]
        va += arr[nt:nt + nv]
        tr += arr[nt + nv:]
    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)
    return tr, va, te


def pack(rows, v2i, head, label_map, extra=None):
    x = torch.tensor([encode(n, v2i) for n, _ in rows], dtype=torch.long)
    y = torch.tensor([label_map[l] for _, l in rows], dtype=torch.long)
    t = {head: y}
    if extra:
        t.update(extra(rows))
    return {"x": x, "targets": t}


def evaluate(model, data, head, ncls, device):
    model.eval()
    with torch.no_grad():
        logits = model(data["x"].to(device))[head]
        pred = logits.argmax(1).cpu()
        y = data["targets"][head]
    acc = (pred == y).float().mean().item()
    rec = []
    for c in range(ncls):
        m = y == c
        if m.sum().item():
            rec.append(((pred == c) & m).float().sum().item() / m.sum().item())
    return acc, float(np.mean(rec)) if rec else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "web" / "model.json"))
    ap.add_argument("--corpus-root", default=r"D:\names-corpus")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    global CORPUS
    CORPUS = Path(args.corpus_root)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    gender_zh = load_gender_zh()
    gender_foreign = load_gender_foreign()
    expression = dedupe(load_jsonl_rows(ROOT / "data" / "moe_expression_v4.jsonl"))
    yuri_chars = dedupe(load_jsonl_rows(ROOT / "data" / "moe_yuri_chars.jsonl")
                        + load_jsonl_rows(ROOT / "data" / "moe_char_negatives.jsonl"))
    genre = dedupe(load_jsonl_rows(ROOT / "data" / "names_clean.jsonl") if False else [])
    if not genre:
        with open(ROOT / "data" / "names_clean.jsonl", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                n = normalize_name(o["name"])
                if valid_name(n) and o["label"] in ("baihe", "purelove", "romance"):
                    genre.append((n, o["label"]))
        genre = dedupe(genre)
    print("expression", Counter(l for _, l in expression), flush=True)
    print("yuri_chars", Counter(l for _, l in yuri_chars), flush=True)
    print("genre", Counter(l for _, l in genre), flush=True)

    gmap = {c: i for i, c in enumerate(HEADS["gender"]["classes"])}
    emap = {c: i for i, c in enumerate(HEADS["expression"]["classes"])}
    ycmap = {c: i for i, c in enumerate(HEADS["yuri_char"]["classes"])}
    jmap = {c: i for i, c in enumerate(HEADS["genre"]["classes"])}
    ymap = {"not_baihe": 0, "baihe": 1}

    def yuri_extra(rows):
        return {"yuri": torch.tensor([ymap["baihe" if l == "baihe" else "not_baihe"] for _, l in rows], dtype=torch.long)}

    datasets = {}
    for tag, rows, head, lmap, extra in [
        ("gender_zh", gender_zh, "gender", gmap, None),
        ("gender_foreign", gender_foreign, "gender", gmap, None),
        ("expression", expression, "expression", emap, None),
        ("yuri_char", yuri_chars, "yuri_char", ycmap, None),
        ("genre", genre, "genre", jmap, yuri_extra),
    ]:
        tr, va, te = split(rows, seed=args.seed)
        tr_aug = augment(tr)
        datasets[tag] = {
            "train": pack(tr_aug, None or {}, head, lmap, extra) if False else None,
            "val": pack(va, {}, head, lmap, extra) if False else None,
            "rows": (tr, va, te),
            "head": head,
        }
        print(tag, "train", len(tr), "->aug", len(tr_aug), "val", len(va), "test", len(te), flush=True)

    all_names = [n for n, _ in gender_zh + gender_foreign + expression + yuri_chars + genre]
    vocab_names = list(all_names) + ["0123456789_", "xX_"]
    for i, n in enumerate(all_names):
        if i % 3 == 0:
            vocab_names.append(n + SUFFIXES[i % len(SUFFIXES)])
    vocab, v2i = build_vocab(vocab_names)
    V = len(vocab)
    print("vocab", V, flush=True)
    for tag, ds in datasets.items():
        tr, va, te = ds["rows"]
        head = ds["head"]
        lmap = {"gender": gmap, "expression": emap, "yuri_char": ycmap, "genre": jmap}[head]
        extra = yuri_extra if head == "genre" else None
        ds["train"] = pack(augment(tr), v2i, head, lmap, extra)
        ds["val"] = pack(va, v2i, head, lmap, extra)
        ds["test"] = pack(te, v2i, head, lmap, extra)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    model = MultiHead(V, drop=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-4)
    losses = {k: nn.CrossEntropyLoss(label_smoothing=0.05) for k in HEADS}
    expr_counts = Counter(l for _, l in expression)
    expr_w = torch.tensor([max(expr_counts.values()) / max(1, expr_counts[c]) for c in HEADS["expression"]["classes"]]).to(device)
    losses["expression"] = nn.CrossEntropyLoss(weight=expr_w, label_smoothing=0.05)
    head_weight = {"gender": 1.0, "expression": 2.0, "yuri_char": 1.5, "genre": 1.0, "yuri": 0.6}
    repeat = {"gender_zh": 1, "gender_foreign": 1, "expression": 8, "yuri_char": 4, "genre": 1}

    batch_list = []
    for tag, ds in datasets.items():
        x, t = ds["train"]["x"], ds["train"]["targets"]
        order = torch.randperm(len(x))
        for i in range(0, len(x), 128):
            idx = order[i:i + 128]
            batch_list.append((tag, x[idx], {k: v[idx] for k, v in t.items()}))

    def val_metrics():
        out = {}
        for tag, ds in datasets.items():
            head = ds["head"]
            ncls = len(HEADS[head]["classes"])
            out[tag] = (head,) + evaluate(model, ds["val"], head, ncls, device)
        out["yuri"] = ("yuri",) + evaluate(model, datasets["genre"]["val"], "yuri", 2, device)
        return out

    best_score, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        random.shuffle(batch_list)
        total = 0.0
        for tag, xb, targets in batch_list:
            for _ in range(repeat[tag]):
                opt.zero_grad()
                out = model(xb.to(device))
                loss = None
                for head, y in targets.items():
                    l = losses[head](out[head], y.to(device)) * head_weight[head]
                    loss = l if loss is None else loss + l
                loss.backward()
                opt.step()
                total += loss.item()
        model.eval()
        m = val_metrics()
        score = float(np.mean([
            m["gender_zh"][2], m["gender_foreign"][2], m["expression"][2],
            m["yuri_char"][2], m["genre"][1], m["yuri"][2],
        ]))
        if ep % 2 == 0 or ep == 1:
            print(f"ep {ep:2d} loss={total/len(batch_list):.4f} " +
                  " ".join(f"{t}:{a:.3f}/{b:.3f}" for t, (h, a, b) in m.items()) +
                  f" score={score:.4f}", flush=True)
        if score > best_score + 1e-5:
            best_score, best_ep, bad = score, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience:
            print("early stop", ep, "best", best_ep, flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    metrics_out = {}
    for tag, ds in datasets.items():
        head = ds["head"]
        acc, bal = evaluate(model, ds["test"], head, len(HEADS[head]["classes"]), device)
        metrics_out[tag] = {"acc": round(acc, 4), "balanced": round(bal, 4), "n": len(ds["test"]["x"])}
        print("TEST", tag, metrics_out[tag], flush=True)
    yacc, ybal = evaluate(model, datasets["genre"]["test"], "yuri", 2, device)
    metrics_out["yuri"] = {"acc": round(yacc, 4), "balanced": round(ybal, 4), "n": len(datasets["genre"]["test"]["x"])}
    print("TEST yuri", metrics_out["yuri"], flush=True)
    baseline = max(Counter(l for _, l in genre).values()) / len(genre)

    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    convs = []
    in_ch = EMB_DIM + POS_DIM
    for i, (k, c) in enumerate(CONVS):
        w = np.transpose(sd[f"convs.{i}.weight"], (0, 2, 1)).reshape(c, k * in_ch)
        convs.append({
            "k": k,
            "W": {"shape": [c, k * in_ch], "data": [round(float(x), 6) for x in w.reshape(-1)]},
            "b": {"shape": [c], "data": [round(float(x), 6) for x in sd[f"convs.{i}.bias"]]},
            "bn": {
                "gamma": {"shape": [c], "data": [round(float(x), 6) for x in sd[f"bns.{i}.weight"]]},
                "beta": {"shape": [c], "data": [round(float(x), 6) for x in sd[f"bns.{i}.bias"]]},
                "mean": {"shape": [c], "data": [round(float(x), 6) for x in sd[f"bns.{i}.running_mean"]]},
                "var": {"shape": [c], "data": [round(float(x), 6) for x in sd[f"bns.{i}.running_var"]]},
            },
        })
    heads_w = {}
    for k in HEADS:
        heads_w[k] = {
            "W": {"shape": [len(HEADS[k]["classes"]), HIDDEN],
                  "data": [round(float(x), 6) for x in sd[f"heads.{k}.weight"].reshape(-1)]},
            "b": {"shape": [len(HEADS[k]["classes"])],
                  "data": [round(float(x), 6) for x in sd[f"heads.{k}.bias"]]},
        }
    fc_in = 2 * sum(c for _, c in CONVS) + 2 * EMB_DIM
    model_json = {
        "format": "baihe-multi-v4",
        "config": {
            "max_len": MAX_LEN, "emb_dim": EMB_DIM, "pos_dim": POS_DIM,
            "conv": [{"k": k, "ch": c} for k, c in CONVS],
            "hidden": HIDDEN, "bn_eps": BN_EPS, "pad_id": PAD_ID, "unk_id": UNK_ID,
            "heads": HEADS,
        },
        "vocab": vocab,
        "weights": {
            "emb": {"shape": [V, EMB_DIM], "data": [round(float(x), 6) for x in sd["emb.weight"].reshape(-1)]},
            "pos": {"shape": [MAX_LEN, POS_DIM], "data": [round(float(x), 6) for x in sd["pos.weight"].reshape(-1)]},
            "conv": convs,
            "fc": {"W": {"shape": [HIDDEN, fc_in], "data": [round(float(x), 6) for x in sd["fc.weight"].reshape(-1)]},
                   "b": {"shape": [HIDDEN], "data": [round(float(x), 6) for x in sd["fc.bias"]]}},
            "heads": heads_w,
        },
        "metrics": {
            "gender": {"acc": metrics_out["gender_zh"]["acc"], "balanced_acc": metrics_out["gender_zh"]["balanced"],
                       "n": metrics_out["gender_zh"]["n"], "acc_zh": metrics_out["gender_zh"]["acc"],
                       "acc_foreign": metrics_out["gender_foreign"]["acc"],
                       "n_zh": metrics_out["gender_zh"]["n"], "n_foreign": metrics_out["gender_foreign"]["n"]},
            "expression": {"acc": metrics_out["expression"]["acc"], "balanced_acc": metrics_out["expression"]["balanced"],
                           "n": metrics_out["expression"]["n"]},
            "yuri_char": {"acc": metrics_out["yuri_char"]["acc"], "balanced_acc": metrics_out["yuri_char"]["balanced"],
                          "n": metrics_out["yuri_char"]["n"]},
            "genre": {"acc": metrics_out["genre"]["acc"], "baseline": round(baseline, 4), "n": metrics_out["genre"]["n"]},
            "yuri": {"balanced_acc": metrics_out["yuri"]["balanced"], "acc": metrics_out["yuri"]["acc"],
                     "n": metrics_out["yuri"]["n"]},
        },
        "dataset": {
            "gender_zh": len(gender_zh), "gender_foreign": len(gender_foreign),
            "expression": len(expression), "yuri_char": len(yuri_chars), "genre": len(genre),
            "note": "真实姓名语料(萌名) + 萌娘百科角色分类 + 晋江文学城作品库",
        },
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(model_json, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("saved", out_path, f"{out_path.stat().st_size/1024/1024:.2f} MB", flush=True)

    train_dir = ROOT / "train"
    splits_out = {tag: {"val": [[n, l] for n, l in ds["rows"][1]], "test": [[n, l] for n, l in ds["rows"][2]]}
                  for tag, ds in datasets.items()}
    (train_dir / "splits_v4.json").write_text(json.dumps(splits_out, ensure_ascii=False), encoding="utf-8")
    (train_dir / "metrics_v4.json").write_text(json.dumps(metrics_out, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# v4 五头模型训练报告

- 性别头：中文 {metrics_out['gender_zh']['acc']:.2%} / 外国名 {metrics_out['gender_foreign']['acc']:.2%}
- ACG 性别表达头（{len(expression)} 条）：acc {metrics_out['expression']['acc']:.2%} / 平衡 {metrics_out['expression']['balanced']:.2%}
- ACG 百合角色头（{len(yuri_chars)} 条，萌娘百科分类:百合 vs 普通角色）：acc {metrics_out['yuri_char']['acc']:.2%} / 平衡 {metrics_out['yuri_char']['balanced']:.2%}
- 作品频道头（{len(genre)} 条）：acc {metrics_out['genre']['acc']:.2%}（基线 {baseline:.2%}）
- 百合判定头：平衡 {metrics_out['yuri']['balanced']:.2%}
- 训练支持网名增广（数字/下划线后缀），最佳 epoch {best_ep}

> 纯娱乐；性取向不针对真人，ACG 角色标签来自公开分类页。
"""
    (train_dir / "train_report_v4.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()

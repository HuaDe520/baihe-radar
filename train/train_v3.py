# -*- coding: utf-8 -*-
"""v3 四头模型训练：性别（真实姓名语料，含外国名）+ ACG 性别表达（萌娘百科角色分类）
+ 作品频道三分类 + 百合判定（晋江小说主角名）。

用法：python train/train_v3.py --out web/model.json
输出：模型 JSON、train/metrics_v3.json、train/splits_v3.json、train/train_report_v3.md
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
REFDATA = Path(r"D:\dsh\_refdata")

MAX_LEN = 12
EMB_DIM = 32
POS_DIM = 8
CONVS = [(1, 64), (2, 96), (3, 96)]
HIDDEN = 160
PAD_ID, UNK_ID = 0, 1
BN_EPS = 1e-5
SEED = 42

HEADS = {
    "gender": {"classes": ["female", "male"], "names_zh": ["女", "男"]},
    "expression": {"classes": ["trap", "futanari", "tomboy", "normal"],
                   "names_zh": ["伪娘", "扶她", "伪郎", "普通"]},
    "genre": {"classes": ["baihe", "purelove", "romance"],
              "names_zh": ["百合", "纯爱(BL)", "言情(BG)"]},
    "yuri": {"classes": ["not_baihe", "baihe"], "names_zh": ["非百合", "百合"]},
}


# ---------------- 数据读取 ----------------

def normalize_name(raw):
    s = "".join(ch for ch in raw.strip() if not ch.isspace())
    return "".join(ch.lower() if "A" <= ch <= "Z" or "a" <= ch <= "z" else ch for ch in s)


def valid_name(name, min_len=2, max_len=MAX_LEN):
    if not (min_len <= len(name) <= max_len):
        return False
    return bool(re.fullmatch(r"[\u4e00-\u9fa5a-z·\-\.]+", name))


def load_gender_zh(limit_per_class=20000):
    path = CORPUS / "Chinese_Names_Corpus" / "Chinese_Names_Corpus_Gender（120W）.txt"
    male, female = [], []
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f):
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            name, sex = parts[0].strip(), parts[1].strip()
            name = normalize_name(name)
            if not valid_name(name) or not re.fullmatch(r"[\u4e00-\u9fa5·]+", name):
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
    print(f"gender_zh: male_pool={len(male)} female_pool={len(female)} used={len(rows)}", flush=True)
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
            for variant in (normalize_name(cn), re.sub(r"[^a-z]", "", en.lower())):
                if valid_name(variant) and (re.search(r"[\u4e00-\u9fa5]", variant) or re.fullmatch(r"[a-z]{2,12}", variant)):
                    (male if label == "male" else female).append(variant)
    rng = random.Random(SEED + 1)
    rng.shuffle(male)
    rng.shuffle(female)
    rows = [(n, "male") for n in male[:limit_per_class]] + [(n, "female") for n in female[:limit_per_class]]
    rng.shuffle(rows)
    print(f"gender_foreign: male_pool={len(male)} female_pool={len(female)} used={len(rows)}", flush=True)
    return rows


def load_expression():
    rows = []
    for path in [REFDATA / "moe_expr.jsonl", REFDATA / "moe_negatives.jsonl"]:
        if not path.exists():
            print("missing", path, flush=True)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                name = normalize_name(o["name"])
                if valid_name(name):
                    rows.append((name, o["label"]))
    # 跨类同名剔除
    by = defaultdict(set)
    for n, l in rows:
        by[n].add(l)
    rows = [(n, l) for n, l in rows if len(by[n]) == 1]
    seen, out = set(), []
    for n, l in rows:
        if (n, l) in seen:
            continue
        seen.add((n, l))
        out.append((n, l))
    print("expression:", Counter(l for _, l in out), flush=True)
    return out


def load_genre():
    rows = []
    with open(ROOT / "data" / "names_clean.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            name = normalize_name(o["name"])
            if valid_name(name) and o["label"] in ("baihe", "purelove", "romance"):
                rows.append((name, o["label"]))
    seen, out = set(), []
    for n, l in rows:
        if (n, l) in seen:
            continue
        seen.add((n, l))
        out.append((n, l))
    print("genre:", Counter(l for _, l in out), flush=True)
    return out


# ---------------- 编码与模型 ----------------

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


# ---------------- 数据打包 ----------------

def pack(rows, vocab, head, label_map, extra=None):
    x = torch.tensor([encode(n, vocab) for n, _ in rows], dtype=torch.long)
    y = torch.tensor([label_map[l] for _, l in rows], dtype=torch.long)
    out = {"x": x, "targets": {head: y}}
    if extra:
        out["targets"].update(extra(rows))
    return out


def split(rows, seed=SEED, ratios=(0.8, 0.1, 0.1)):
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


def evaluate_head(model, data, head, ncls, device):
    model.eval()
    with torch.no_grad():
        logits = model(data["x"].to(device))[head]
        pred = logits.argmax(1).cpu()
        y = data["targets"][head]
    acc = (pred == y).float().mean().item()
    recalls = []
    for c in range(ncls):
        mask = y == c
        if mask.sum().item() > 0:
            recalls.append(((pred == c) & mask).float().sum().item() / mask.sum().item())
    return acc, float(np.mean(recalls)) if recalls else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "web" / "model.json"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--corpus-root", default=r"D:\names-corpus")
    ap.add_argument("--refdata-root", default=r"D:\dsh\_refdata")
    args = ap.parse_args()
    global CORPUS, REFDATA
    CORPUS = Path(args.corpus_root)
    REFDATA = Path(args.refdata_root)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    gender_zh = load_gender_zh()
    gender_foreign = load_gender_foreign()
    expression = load_expression()
    genre = load_genre()
    all_names = [n for n, _ in gender_zh + gender_foreign + expression + genre]
    vocab, v2i = build_vocab(all_names)
    V = len(vocab)
    print("vocab", V, flush=True)

    gmap = {c: i for i, c in enumerate(HEADS["gender"]["classes"])}
    emap = {c: i for i, c in enumerate(HEADS["expression"]["classes"])}
    jmap = {c: i for i, c in enumerate(HEADS["genre"]["classes"])}
    ymap = {"not_baihe": 0, "baihe": 1}

    def yuri_extra(rows):
        return {"yuri": torch.tensor([ymap["baihe" if l == "baihe" else "not_baihe"] for _, l in rows], dtype=torch.long)}

    datasets = {}
    for tag, rows, head, lmap in [
        ("gender_zh", gender_zh, "gender", gmap),
        ("gender_foreign", gender_foreign, "gender", gmap),
        ("expression", expression, "expression", emap),
        ("genre", genre, "genre", jmap),
    ]:
        tr, va, te = split(rows, seed=args.seed)
        datasets[tag] = {
            "train": pack(tr, v2i, head, lmap, yuri_extra if head == "genre" else None),
            "val": pack(va, v2i, head, lmap, yuri_extra if head == "genre" else None),
            "test": pack(te, v2i, head, lmap, yuri_extra if head == "genre" else None),
            "head": head,
            "sizes": (len(tr), len(va), len(te)),
            "rows": (tr, va, te),
        }
        print(tag, "sizes", datasets[tag]["sizes"], flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    model = MultiHead(V, drop=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-4)

    # 表达类样本少 → 类别权重 + 过采样
    expr_counts = Counter(l for _, l in expression)
    expr_w = torch.tensor([max(expr_counts.values()) / max(1, expr_counts[c]) for c in HEADS["expression"]["classes"]]).to(device)
    losses = {
        "gender": nn.CrossEntropyLoss(label_smoothing=0.05),
        "expression": nn.CrossEntropyLoss(weight=expr_w, label_smoothing=0.05),
        "genre": nn.CrossEntropyLoss(label_smoothing=0.05),
        "yuri": nn.CrossEntropyLoss(label_smoothing=0.05),
    }
    head_weight = {"gender": 1.0, "expression": 2.0, "genre": 1.0, "yuri": 0.6}
    # 每个数据集每轮重复次数（控制采样）
    repeat = {"gender_zh": 1, "gender_foreign": 1, "expression": 10, "genre": 1}

    batch_list = []
    for tag, ds in datasets.items():
        x, targets = ds["train"]["x"], ds["train"]["targets"]
        n = len(x)
        order = torch.randperm(n)
        for i in range(0, n, 128):
            idx = order[i:i + 128]
            batch_list.append((tag, x[idx], {k: v[idx] for k, v in targets.items()}))

    def all_val_metrics(tag):
        ds = datasets[tag]
        head = ds["head"]
        ncls = len(HEADS[head]["classes"])
        acc, bal = evaluate_head(model, ds["val"], head, ncls, device)
        return head, acc, bal

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
        scores = []
        metrics = {}
        for tag in datasets:
            head, acc, bal = all_val_metrics(tag)
            metrics[tag] = (head, acc, bal)
            if head == "gender":
                scores.append(bal)
            elif head == "expression":
                scores.append(bal)
            elif head == "genre":
                scores.append(acc)
            elif head == "yuri":
                pass
        # yuri 单独评估
        yacc, ybal = evaluate_head(model, datasets["genre"]["val"], "yuri", 2, device)
        metrics["yuri"] = ("yuri", yacc, ybal)
        scores.append(ybal)
        score = float(np.mean(scores))
        if ep % 2 == 0 or ep == 1:
            print(f"ep {ep:2d} loss={total/len(batch_list):.4f} " +
                  " ".join(f"{t}:{a:.3f}/{b:.3f}" for t, (h, a, b) in metrics.items()) +
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

    # 测试集指标
    metrics_out = {}
    for tag in ["gender_zh", "gender_foreign", "expression", "genre"]:
        acc, bal = evaluate_head(model, datasets[tag]["test"], datasets[tag]["head"],
                                 len(HEADS[datasets[tag]["head"]]["classes"]), device)
        metrics_out[tag] = {"acc": round(acc, 4), "balanced": round(bal, 4), "n": len(datasets[tag]["test"]["x"])}
        print("TEST", tag, metrics_out[tag], flush=True)
    yacc, ybal = evaluate_head(model, datasets["genre"]["test"], "yuri", 2, device)
    metrics_out["yuri"] = {"acc": round(yacc, 4), "balanced": round(ybal, 4), "n": len(datasets["genre"]["test"]["x"])}
    print("TEST yuri", metrics_out["yuri"], flush=True)
    gen_acc = metrics_out["genre"]["acc"]
    baseline = max(Counter(l for _, l in genre).values()) / len(genre)
    print("genre baseline", round(baseline, 4), flush=True)

    # 导出
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    convs = []
    in_ch = EMB_DIM + POS_DIM
    for i, (k, c) in enumerate(CONVS):
        w = sd[f"convs.{i}.weight"]
        w2 = np.transpose(w, (0, 2, 1)).reshape(c, k * in_ch)
        convs.append({
            "k": k,
            "W": {"shape": [c, k * in_ch], "data": [round(float(x), 6) for x in w2.reshape(-1)]},
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
        "format": "baihe-multi-v3",
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
            "fc": {"W": {"shape": [HIDDEN, fc_in],
                         "data": [round(float(x), 6) for x in sd["fc.weight"].reshape(-1)]},
                   "b": {"shape": [HIDDEN],
                         "data": [round(float(x), 6) for x in sd["fc.bias"]]}},
            "heads": heads_w,
        },
        "metrics": {
            "gender": {"acc": metrics_out["gender_zh"]["acc"],
                       "balanced_acc": metrics_out["gender_zh"]["balanced"],
                       "n": metrics_out["gender_zh"]["n"],
                       "acc_zh": metrics_out["gender_zh"]["acc"],
                       "acc_foreign": metrics_out["gender_foreign"]["acc"],
                       "n_zh": metrics_out["gender_zh"]["n"],
                       "n_foreign": metrics_out["gender_foreign"]["n"]},
            "expression": {"acc": metrics_out["expression"]["acc"],
                           "balanced_acc": metrics_out["expression"]["balanced"],
                           "n": metrics_out["expression"]["n"]},
            "genre": {"acc": gen_acc, "baseline": round(baseline, 4), "n": metrics_out["genre"]["n"]},
            "yuri": {"balanced_acc": metrics_out["yuri"]["balanced"], "acc": metrics_out["yuri"]["acc"],
                     "n": metrics_out["yuri"]["n"]},
        },
        "dataset": {
            "gender_zh": len(gender_zh), "gender_foreign": len(gender_foreign),
            "expression": len(expression), "genre": len(genre),
            "source": "真实姓名语料库(萌名) + 萌娘百科角色分类 + 晋江文学城作品库",
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model_json, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("saved", out_path, f"{out_path.stat().st_size/1024/1024:.2f} MB", flush=True)

    # splits + metrics + report
    train_dir = ROOT / "train"
    splits_out = {
        tag: {"val": [[n, l] for n, l in ds["rows"][1]],
              "test": [[n, l] for n, l in ds["rows"][2]]}
        for tag, ds in datasets.items()
    }
    (train_dir / "splits_v3.json").write_text(json.dumps(splits_out, ensure_ascii=False), encoding="utf-8")
    (train_dir / "metrics_v3.json").write_text(json.dumps(metrics_out, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# v3 四头模型训练报告

- 性别头：真实姓名语料（中文 {len(gender_zh)} 条 + 外国名/音译 {len(gender_foreign)} 条），
  中文测试 acc {metrics_out['gender_zh']['acc']:.2%}（平衡 {metrics_out['gender_zh']['balanced']:.2%}），
  外国名测试 acc {metrics_out['gender_foreign']['acc']:.2%}（平衡 {metrics_out['gender_foreign']['balanced']:.2%}）
- ACG 性别表达头（伪娘/扶她/伪郎/普通，萌娘百科角色分类 {len(expression)} 条）：
  测试 acc {metrics_out['expression']['acc']:.2%}（平衡 {metrics_out['expression']['balanced']:.2%}）
- 作品频道头（晋江小说主角名 {len(genre)} 条）：测试 acc {gen_acc:.2%}（多数类基线 {baseline:.2%}）
- 百合判定头：平衡准确率 {metrics_out['yuri']['balanced']:.2%}
- 最佳 epoch：{best_ep}

> 性别头来自真实姓名的统计用字规律，只是「名字像男/像女」的概率，不是身份判断；
> 性取向不针对真实个人做预测。ACG 属性只对二次元角色名有参考价值。纯娱乐。
"""
    (train_dir / "train_report_v3.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()

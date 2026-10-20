# -*- coding: utf-8 -*-
"""用公开数据估计「取向套娃」的 4×4 转移矩阵（数据估计 + 弱内置先验平滑）。

数据：
1) 晋江作品库「主角」配对（≥2 个名字）：百合→女同、纯爱→男同、言情→按性别头预测男女；
   清洗：言情只保留预测为一男一女，百合/纯爱只保留预测同性的配对。
2) VNDB：角色所属 VN 的标签（Yuri/Lesbian vs BL vs Romance）给角色定类型，同一 VN 角色构成边。

输出：web/orientation_priors.json
用法：python train/learn_orientation_priors.py [--model web/model.json]
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from ref_predict_v4 import load_model, predict  # noqa: E402

TYPES = ["straight_male", "straight_female", "gay_male", "lesbian"]
IDX = {t: i for i, t in enumerate(TYPES)}
ALPHA = 0.5
PRIOR_STRENGTH = 300.0
HAND = [
    [0.15, 0.60, 0.10, 0.15],
    [0.60, 0.15, 0.15, 0.10],
    [0.30, 0.15, 0.40, 0.15],
    [0.15, 0.30, 0.15, 0.40],
]
YURI_WORDS = ("yuri", "lesbian", "female love", "girls love", "shoujo ai", "girls' love")
BL_WORDS = ("boys love", "boy's love", "boys' love", "male on male", "gay", "shounen ai", "bara", "bl")


def gender_of(model, name):
    p = predict(model, name)
    return "female" if p["gender"][0] >= p["gender"][1] else "male"


def add_pair(counts, a, b):
    counts[IDX[a]][IDX[b]] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "web" / "model.json"))
    ap.add_argument("--books", default=str(ROOT / "data" / "_books.jsonl"))
    ap.add_argument("--vndb", default=str(ROOT / "data" / "raw" / "vndb_chars.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "web" / "orientation_priors.json"))
    args = ap.parse_args()

    model = load_model(args.model)
    counts = [[0.0] * 4 for _ in range(4)]
    n_pairs = 0

    books_path = Path(args.books)
    books = [json.loads(l) for l in open(books_path, encoding="utf-8") if l.strip()] if books_path.exists() else []
    print("books", len(books), flush=True)
    cache = {}
    for b in books:
        names = b.get("names") or []
        if len(names) < 2:
            continue
        label = b.get("label")
        genders = []
        for n in names[:3]:
            if n not in cache:
                try:
                    cache[n] = gender_of(model, n)
                except Exception:
                    cache[n] = "female"
            genders.append(cache[n])
        if label == "romance":
            if not ("female" in genders and "male" in genders):
                continue
        elif len(set(genders)) != 1:
            continue
        types = []
        for g in genders:
            if label == "baihe":
                types.append("lesbian")
            elif label == "purelove":
                types.append("gay_male")
            else:
                types.append("straight_female" if g == "female" else "straight_male")
        for i in range(len(types)):
            for j in range(len(types)):
                if i != j:
                    add_pair(counts, types[i], types[j])
                    n_pairs += 1

    vndb_path = Path(args.vndb)
    vn_chars = {}
    if vndb_path.exists():
        for line in open(vndb_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            ch = json.loads(line)
            g = (ch.get("gender") or [None])[0]
            if g not in ("m", "f"):
                continue
            for vn in ch.get("vns") or []:
                tags = [t.get("name", "") for t in vn.get("tags") or []]
                vn_chars.setdefault(vn["id"], []).append((g, tags))

    def vn_kind(tags_list):
        txt = " ".join(" ".join(t) for t in tags_list).lower()
        if any(w in txt for w in YURI_WORDS):
            return "lesbian"
        if any(w in txt for w in BL_WORDS):
            return "gay_male"
        if "romance" in txt:
            return "straight"
        return None

    for vid, entries in vn_chars.items():
        kind = vn_kind([tags for _, tags in entries])
        if not kind:
            continue
        chars = [g for g, _ in entries][:6]
        for i in range(len(chars)):
            for j in range(len(chars)):
                if i == j:
                    continue
                a, b = chars[i], chars[j]
                if kind == "lesbian" and a == "f" and b == "f":
                    add_pair(counts, "lesbian", "lesbian")
                    n_pairs += 1
                elif kind == "gay_male" and a == "m" and b == "m":
                    add_pair(counts, "gay_male", "gay_male")
                    n_pairs += 1
                elif kind == "straight":
                    ta = "straight_female" if a == "f" else "straight_male"
                    tb = "straight_female" if b == "f" else "straight_male"
                    if ta != tb:
                        add_pair(counts, ta, tb)
                        n_pairs += 1

    matrix_raw, matrix = [], []
    for i in range(4):
        row = [c + ALPHA for c in counts[i]]
        sm = sum(row)
        matrix_raw.append([round(x / sm, 6) for x in row])
        row2 = [counts[i][j] + PRIOR_STRENGTH * HAND[i][j] for j in range(4)]
        s2 = sum(row2)
        matrix.append([round(x / s2, 6) for x in row2])

    out = {
        "format": "orientation-priors-v1",
        "source": "jjwxc 主角配对 + VNDB VN 角色配对",
        "n_pairs": n_pairs,
        "counts": counts,
        "alpha": ALPHA,
        "prior_strength": PRIOR_STRENGTH,
        "prior": "builtin hand matrix (weak prior)",
        "matrix": matrix,
        "matrix_raw": matrix_raw,
        "types": TYPES,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", args.out, "n_pairs", n_pairs, flush=True)
    print("matrix", matrix, flush=True)


if __name__ == "__main__":
    main()

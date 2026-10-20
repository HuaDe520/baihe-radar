# -*- coding: utf-8 -*-
"""v4 五头模型 numpy 参考推理（与 web/inference.js 对齐）。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "web" / "model.json"
HEADS = ["gender", "expression", "yuri_char", "genre", "yuri"]


def load_model(path=None):
    with open(path or DEFAULT_MODEL, encoding="utf-8") as f:
        return json.load(f)


def normalize_input(raw):
    s = "".join(ch for ch in raw.strip() if not ch.isspace())
    return "".join(ch.lower() if "A" <= ch <= "Z" else ch for ch in s)


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def predict(model, raw):
    cfg = model["config"]
    W = model["weights"]
    name = normalize_input(raw)
    if not name:
        raise ValueError("empty input")
    chars = list(name)[: cfg["max_len"]]
    v2i = {ch: i for i, ch in enumerate(model["vocab"])}
    ids = [v2i.get(ch, cfg["unk_id"]) for ch in chars]
    ids += [cfg["pad_id"]] * (cfg["max_len"] - len(ids))

    emb = np.asarray(W["emb"]["data"], dtype=np.float64).reshape(W["emb"]["shape"])
    pos = np.asarray(W["pos"]["data"], dtype=np.float64).reshape(W["pos"]["shape"])
    X = np.concatenate([emb[ids], pos], axis=1)
    in_ch = cfg["emb_dim"] + cfg["pos_dim"]

    pooled = []
    for br in W["conv"]:
        k = br["k"]
        Wc = np.asarray(br["W"]["data"], dtype=np.float64).reshape(br["W"]["shape"])
        b = np.asarray(br["b"]["data"], dtype=np.float64)
        bn = br["bn"]
        g = np.asarray(bn["gamma"]["data"], dtype=np.float64)
        be = np.asarray(bn["beta"]["data"], dtype=np.float64)
        mu = np.asarray(bn["mean"]["data"], dtype=np.float64)
        var = np.asarray(bn["var"]["data"], dtype=np.float64)
        zs = [b + Wc @ X[p:p + k].reshape(-1) for p in range(cfg["max_len"] - k + 1)]
        z = np.stack(zs, 0)
        zn = (z - mu) / np.sqrt(var + cfg["bn_eps"]) * g + be
        a = np.maximum(0.0, zn)
        pooled.append(a.max(0))
        pooled.append(a.mean(0))
    last = max(1, sum(1 for i in ids if i != cfg["pad_id"]))
    h0 = np.concatenate(pooled + [emb[ids[0]], emb[ids[last - 1]]])

    fc = W["fc"]
    h = np.maximum(0.0, np.asarray(fc["W"]["data"]).reshape(fc["W"]["shape"]) @ h0
                   + np.asarray(fc["b"]["data"]))
    probs = {}
    for head in HEADS:
        hw = W["heads"][head]
        logits = np.asarray(hw["W"]["data"]).reshape(hw["W"]["shape"]) @ h + np.asarray(hw["b"]["data"])
        probs[head] = [round(float(x), 6) for x in _softmax(logits)]
    return probs


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else None
    names = sys.argv[2:] if model_path else sys.argv[1:]
    if not names:
        names = ["林晚", "张伟", "Alice", "林晚_233"]
    model = load_model(model_path)
    for n in names:
        p = predict(model, n)
        print(n, {k: [round(x, 3) for x in v] for k, v in p.items()})


if __name__ == "__main__":
    main()

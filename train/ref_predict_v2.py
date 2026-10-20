# -*- coding: utf-8 -*-
"""v2 双头模型 numpy 参考推理（严格按 CONTRACT v2 算法），用于与 JS 推理对拍。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "_work" / "model_v2.json"


def load_model(path=None):
    p = Path(path) if path else DEFAULT_MODEL
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def predict(model, raw):
    cfg = model["config"]
    W = model["weights"]
    v2i = {ch: i for i, ch in enumerate(model["vocab"])}
    chars = [ch for ch in raw if not ch.isspace()][: cfg["max_len"]]
    ids = [v2i.get(ch, cfg["unk_id"]) for ch in chars]
    ids += [cfg["pad_id"]] * (cfg["max_len"] - len(ids))

    emb = np.asarray(W["emb"]["data"], dtype=np.float64).reshape(W["emb"]["shape"])
    pos = np.asarray(W["pos"]["data"], dtype=np.float64).reshape(W["pos"]["shape"])
    X = np.concatenate([emb[ids], pos], axis=1)  # 8 x 40（emb 与 pos 拼接，不是相加）
    in_ch = cfg["emb_dim"] + cfg["pos_dim"]

    pooled = []
    for br in W["conv"]:
        k = br["k"]
        Wc = np.asarray(br["W"]["data"], dtype=np.float64).reshape(br["W"]["shape"])
        b = np.asarray(br["b"]["data"], dtype=np.float64)
        bn = br["bn"]
        gamma = np.asarray(bn["gamma"]["data"], dtype=np.float64)
        beta = np.asarray(bn["beta"]["data"], dtype=np.float64)
        mean = np.asarray(bn["mean"]["data"], dtype=np.float64)
        var = np.asarray(bn["var"]["data"], dtype=np.float64)
        zs = []
        for p in range(cfg["max_len"] - k + 1):
            z = b + Wc @ X[p:p + k].reshape(-1)
            zs.append(z)
        z = np.stack(zs, 0)                                    # (positions, C)
        zn = (z - mean) / np.sqrt(var + cfg["bn_eps"]) * gamma + beta
        a = np.maximum(0.0, zn)
        pooled.append(a.max(0))
        pooled.append(a.mean(0))
    lens = sum(1 for i in ids if i != cfg["pad_id"])
    lens = max(lens, 1)
    h0 = np.concatenate(pooled + [emb[ids[0]], emb[ids[lens - 1]]])

    fc = W["fc"]
    h = np.maximum(0.0, np.asarray(fc["W"]["data"]).reshape(fc["W"]["shape"]) @ h0
                   + np.asarray(fc["b"]["data"]))
    o3 = W["out3"]
    logits3 = np.asarray(o3["W"]["data"]).reshape(o3["W"]["shape"]) @ h + np.asarray(o3["b"]["data"])
    o2 = W["out2"]
    logits2 = np.asarray(o2["W"]["data"]).reshape(o2["W"]["shape"]) @ h + np.asarray(o2["b"]["data"])
    return [round(float(x), 6) for x in _softmax(logits3)], [round(float(x), 6) for x in _softmax(logits2)]


def main():
    model = load_model(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else None)
    names = sys.argv[2:] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else sys.argv[1:]
    if not names:
        names = ["沈清和", "林晚", "张伟", "顾言"]
    for n in names:
        p3, p2 = predict(model, n)
        print(n, "p3=", p3, "p2=", p2, "百合指数=", round(p2[1] * 100, 1))


if __name__ == "__main__":
    main()

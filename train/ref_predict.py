# -*- coding: utf-8 -*-
"""纯 numpy 参考推理：严格按 CONTRACT.md 第二节实现，用于和 JS 推理对拍。

用法：
  python train/ref_predict.py 顾清 林晚 ...
  python train/ref_predict.py            # 读 train/test_vectors.json 的输入重算
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "web" / "model.json"


def load_model(path=MODEL):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def predict(model, raw):
    cfg = model["config"]
    v2i = {ch: i for i, ch in enumerate(model["vocab"])}
    chars = [ch for ch in raw if not ch.isspace()][: cfg["max_len"]]
    ids = [v2i.get(ch, cfg["unk_id"]) for ch in chars]
    ids += [cfg["pad_id"]] * (cfg["max_len"] - len(ids))

    W = model["weights"]
    emb = np.asarray(W["emb"]["data"], dtype=np.float64).reshape(W["emb"]["shape"])
    X = emb[ids]  # 8 x 28

    pooled = []
    for br in W["conv"]:
        k = br["k"]
        Wc = np.asarray(br["W"]["data"], dtype=np.float64).reshape(br["W"]["shape"])
        b = np.asarray(br["b"]["data"], dtype=np.float64)
        zs = []
        for p in range(cfg["max_len"] - k + 1):
            zs.append(b + Wc @ X[p:p + k].reshape(-1))
        z = np.stack(zs, 0)          # (positions, C)
        pooled.append(np.maximum(0.0, z.max(0)))
    h0 = np.concatenate(pooled)

    fc = W["fc"]
    h1 = np.maximum(0.0, np.asarray(fc["W"]["data"], dtype=np.float64).reshape(fc["W"]["shape"]) @ h0
                    + np.asarray(fc["b"]["data"], dtype=np.float64))
    o = W["out"]
    logits = np.asarray(o["W"]["data"], dtype=np.float64).reshape(o["W"]["shape"]) @ h1 \
        + np.asarray(o["b"]["data"], dtype=np.float64)
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return [round(float(x), 6) for x in probs]


def main():
    model = load_model()
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            print(name, predict(model, name))
        return
    # 兜底：读 test_vectors
    tv = ROOT / "train" / "test_vectors.json"
    if tv.exists():
        data = json.loads(tv.read_text(encoding="utf-8"))
        worst = 0.0
        for case in data["cases"]:
            p = predict(model, case["input"])
            d = max(abs(a - b) for a, b in zip(p, case["probs"]))
            worst = max(worst, d)
        print(f"{len(data['cases'])} 条自洽检查，最大差异 {worst:.2e}")


if __name__ == "__main__":
    main()

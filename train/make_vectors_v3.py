# -*- coding: utf-8 -*-
"""生成 train/test_vectors3.json：v3 多头模型 Python 参考推理结果，供 web/parity_test.mjs 对拍。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from ref_predict_v3 import load_model, predict  # noqa: E402

TAKE = {"gender_zh": 80, "gender_foreign": 60, "expression": 25, "genre": 80}
EXTRA = ["张伟", "王芳", "林晚", "沈清和", "顾言", "Alice", "alice", "Bob", "JOHN",
         "雅克·迪吕波", "欧阳娜娜", "A", "一二三四五六七八九十十一", " 张 伟 "]


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else str(ROOT / "web" / "model.json")
    model = load_model(model_path)
    splits = json.loads((ROOT / "train" / "splits_v3.json").read_text(encoding="utf-8"))
    names = []
    for tag, k in TAKE.items():
        test = splits.get(tag, {}).get("test", [])
        names += [x[0] for x in test[:k]]
    names += EXTRA
    cases = [{"input": n, "probs": predict(model, n)} for n in names]
    out = ROOT / "train" / "test_vectors3.json"
    out.write_text(json.dumps({"format": "baihe-multi-v3", "cases": cases}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {out}，共 {len(cases)} 条")


if __name__ == "__main__":
    main()

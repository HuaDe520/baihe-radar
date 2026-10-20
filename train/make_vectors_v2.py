# -*- coding: utf-8 -*-
"""生成 train/test_vectors2.json：v2 双头模型的 Python 参考推理结果，供 web/parity_test.mjs 对拍。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from ref_predict_v2 import load_model, predict  # noqa: E402


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else str(ROOT / "_work" / "model_v2.json")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    model = load_model(model_path)
    splits = json.loads((ROOT / "train" / "splits_v2.json").read_text(encoding="utf-8"))
    cases = []
    names = [x[0] for x in splits["test"]][:n]
    extras = ["张伟", "王芳", "李娜", "林晚", "苏清", "沈月", "顾言", "陈星",
              "欧阳娜娜", "爱丽丝", "A", "一二三四五六七八九十", " 张 伟 "]
    for nm in names + extras:
        p3, p2 = predict(model, nm)
        cases.append({"input": nm, "probs3": p3, "probs2": p2})
    out = ROOT / "train" / "test_vectors2.json"
    out.write_text(json.dumps({"format": "baihe-dualhead-v2", "cases": cases}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {out}，共 {len(cases)} 条")


if __name__ == "__main__":
    main()

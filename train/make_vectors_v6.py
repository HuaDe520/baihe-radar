# -*- coding: utf-8 -*-
"""生成 train/test_vectors4.json：v4 五头模型 Python 参考输出，供 web/parity_test.mjs 对拍。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from ref_predict_v4 import load_model, predict  # noqa: E402

TAKE = {"gender_zh": 50, "gender_foreign": 60, "expression": 25, "yuri_char": 25, "genre": 50}
EXTRA = ["张伟", "王芳", "林晚", "沈清和", "顾言", "Alice", "alice", "Bob", "JOHN",
         "雅克·迪吕波", "欧阳娜娜", "林晚_233", "Alice_777", "xX_张伟_Xx", "名字233",
         "一二三四五六七八九十十一", " 张 伟 ", "フォーニ", "渡良瀬 準", "Watarase Jun"]


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else str(ROOT / "web" / "model.json")
    model = load_model(model_path)
    splits = json.loads((ROOT / "train" / "splits_v6.json").read_text(encoding="utf-8"))
    names = []
    for tag, k in TAKE.items():
        names += [x[0] for x in splits.get(tag, {}).get("test", [])[:k]]
    names += EXTRA
    cases = [{"input": n, "probs": predict(model, n)} for n in names]
    out = ROOT / "train" / "test_vectors4.json"
    out.write_text(json.dumps({"format": "baihe-multi-v4", "cases": cases}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {out}，共 {len(cases)} 条")


if __name__ == "__main__":
    main()

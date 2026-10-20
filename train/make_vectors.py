# -*- coding: utf-8 -*-
"""生成 train/test_vectors.json：Python 参考推理结果，供 web/parity_test.mjs 对拍。

用法：python train/make_vectors.py [N]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from ref_predict import load_model, predict  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    model = load_model()
    splits = json.loads((ROOT / "train" / "splits.json").read_text(encoding="utf-8"))
    test = splits["test"]
    # 测试集名字优先，不足用训练集补
    names = [x[0] for x in test][:n]
    if len(names) < n:
        names += [x[0] for x in splits["train"]][: n - len(names)]
    cases = [{"input": nm, "probs": predict(model, nm)} for nm in names]
    # 再补充几组边界与常见名字，覆盖截断/未知字/单字
    extras = ["张伟", "王芳", "李娜", "林晚", "苏清", "沈月", "顾言", "陈星",
              "欧阳娜娜", "爱丽丝", "A", "一二三四五六七八九十", "", " 张 伟 "]
    for nm in extras:
        if nm == "":
            continue
        cases.append({"input": nm, "probs": predict(model, nm)})
    out = ROOT / "train" / "test_vectors.json"
    out.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {out}，共 {len(cases)} 条")


if __name__ == "__main__":
    main()

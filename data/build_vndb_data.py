# -*- coding: utf-8 -*-
"""从 data/raw/vndb_chars.jsonl 派生训练集（日文名性别 / 表达 / 百合角色）。

用法：python data/build_vndb_data.py
输出：data/vndb_gender.jsonl、data/vndb_expression.jsonl、data/vndb_yuri_chars.jsonl
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "vndb_chars.jsonl"
OUT = ROOT / "data"

JP_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\u30fc々・ーa-z0-9_]+")
YURI_WORDS = ("yuri", "lesbian", "female love", "girls love", "shoujo ai", "girls' love")
BL_WORDS = ("boys love", "boy's love", "boys' love", "male on male", "gay", "shounen ai", "bara", "bl")


def variants(ch):
    out = []
    for raw, is_romaji in ((ch.get("original"), False), (ch.get("name"), True)):
        if not raw:
            continue
        s = "".join(c for c in raw.strip() if not c.isspace())
        if is_romaji:
            s = s.lower()
        if 2 <= len(s) <= 12 and JP_RE.fullmatch(s):
            out.append(s)
    return out


def main():
    if not SRC.exists():
        raise SystemExit(f"缺少 {SRC}，请先运行 data/scrape_vndb.py")
    chars = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    gender_rows, expr_rows, yuri_rows = [], [], []
    for ch in chars:
        gender = (ch.get("gender") or [None])[0]
        traits = set(ch.get("traits") or [])
        vs = variants(ch)
        if not vs:
            continue
        if gender in ("m", "f"):
            lab = "male" if gender == "m" else "female"
            gender_rows += [{"name": v, "label": lab} for v in vs]
        if "trap" in traits or "reverse_trap" in traits:
            expr_rows += [{"name": v, "label": "trap"} for v in vs]
        if "futanari" in traits:
            expr_rows += [{"name": v, "label": "futanari"} for v in vs]
        if "tomboy" in traits:
            expr_rows += [{"name": v, "label": "tomboy"} for v in vs]
        vn_tags = [[t.get("name", "") for t in vn.get("tags") or []] for vn in ch.get("vns") or []]
        is_yuri = any(any(w in t.lower() for w in YURI_WORDS) for tags in vn_tags for t in tags)
        is_bl = any(any(w in t.lower() for w in BL_WORDS) for tags in vn_tags for t in tags)
        if is_yuri and not is_bl:
            yuri_rows += [{"name": v, "label": "yuri_char"} for v in vs]
        elif not is_yuri and not is_bl:
            yuri_rows += [{"name": v, "label": "other_char"} for v in vs]

    pos = {r["name"] for r in expr_rows}
    normal = []
    for ch in chars:
        if set(ch.get("traits") or []) & {"trap", "reverse_trap", "futanari", "tomboy"}:
            continue
        for v in variants(ch):
            if v not in pos:
                normal.append({"name": v, "label": "normal"})
    expr_rows += normal[: max(600, len(expr_rows) // 2)]

    def save(path, rows):
        seen, out = set(), []
        for r in rows:
            key = (r["name"], r["label"])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        with open(path, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(path.name, len(out), Counter(r["label"] for r in out), flush=True)

    save(OUT / "vndb_gender.jsonl", gender_rows)
    save(OUT / "vndb_expression.jsonl", expr_rows)
    save(OUT / "vndb_yuri_chars.jsonl", yuri_rows)


if __name__ == "__main__":
    main()

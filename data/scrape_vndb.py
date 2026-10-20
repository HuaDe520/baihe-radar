# -*- coding: utf-8 -*-
"""抓取 VNDB 角色数据（日文名/罗马音/性别/特征/VN 标签）。

用法：python data/scrape_vndb.py
输出：data/raw/vndb_chars.jsonl（原始，gitignored）；再由 build_vndb_data.py 派生训练集。
说明：VNDB 是公开的视觉小说数据库，API 免费；只取角色公开字段，用于娱乐向名字玩具。
"""
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "vndb_chars.jsonl"
API = "https://api.vndb.org/kana/character"
UA = "baihe-radar/0.1"
TRAITS = {"i51": "trap", "i52": "reverse_trap", "i737": "futanari", "i626": "tomboy", "i48": "crossdress"}


def post(body, retries=5):
    for i in range(retries):
        try:
            req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "User-Agent": UA})
            return json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
        except Exception as e:
            print("retry", i, repr(e)[:100], flush=True)
            time.sleep(1.5 * (i + 1))
    return {}


def fetch(filters, fields, max_pages, tag=None):
    rows = []
    for page in range(1, max_pages + 1):
        j = post({"filters": filters, "fields": fields, "results": 100, "sort": "id", "page": page})
        got = j.get("results") or []
        rows += got
        if page % 5 == 0:
            print(f"{tag or filters} page {page} total {len(rows)}", flush=True)
        if not j.get("more") or not got:
            break
        time.sleep(0.35)
    print(f"{tag or filters} => {len(rows)}", flush=True)
    return rows


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = "id,name,original,gender,vns{id,tags{id,name}}"
    chars = {}
    for r in fetch([], fields, max_pages=25, tag="general"):
        chars[r["id"]] = r
    for tid, label in TRAITS.items():
        pages = 8 if label in ("trap", "crossdress") else 4
        for r in fetch(["trait", "=", tid], "id,name,original,gender", max_pages=pages, tag=label):
            cur = chars.get(r["id"], {"id": r["id"], "traits": []})
            for k in ("name", "original", "gender"):
                if r.get(k):
                    cur[k] = r[k]
            cur.setdefault("traits", [])
            if label not in cur["traits"]:
                cur["traits"].append(label)
            chars[r["id"]] = cur
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in chars.values()), encoding="utf-8")
    print("saved", OUT, len(chars), flush=True)


if __name__ == "__main__":
    main()

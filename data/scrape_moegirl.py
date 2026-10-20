# -*- coding: utf-8 -*-
"""抓取萌娘百科角色分类，构造 ACG 性别表达数据集（正类 + 普通角色负类）。

用法：
  python data/scrape_moegirl.py            # 需要 playwright + 本机 Edge
输出：
  data/moe_expression.jsonl   {"name","label"}  label ∈ trap/futanari/tomboy
  data/moe_negatives.jsonl    {"name","label":"normal"}

注意：萌娘百科有 Cloudflare 校验，脚本用真实浏览器（Playwright + Edge headless）访问；
请控制频率、只抓公开分类页；数据仅用于娱乐向玩具训练。
"""
import asyncio
import json
import re
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
BASE = "https://zh.moegirl.org.cn"
POS_CATS = {"trap": "伪娘", "futanari": "扶她", "tomboy": "伪郎"}
CONCURRENCY = 5


def clean_title(t: str) -> str:
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t.strip())
    return re.sub(r"\s+", "", t)


def valid(t: str) -> bool:
    return bool(t) and 2 <= len(t) <= 10 and bool(re.fullmatch(r"[\u4e00-\u9fa5A-Za-z·\-\.\s\d]+", t))


async def collect_category(ctx, cat, want=600):
    page = await ctx.new_page()
    names = []
    try:
        url = f"{BASE}/Category:{cat}"
        for _ in range(4):
            await page.goto(url, timeout=60000, wait_until="domcontentloaded")
            await page.wait_for_timeout(4000)
            links = await page.eval_on_selector_all("#mw-pages a", "els => els.map(e => e.textContent.trim())")
            names += [clean_title(x) for x in links if valid(clean_title(x))]
            nxt = await page.eval_on_selector_all('a:has-text("下一页")', "els => els.map(e => e.getAttribute('href'))")
            if not nxt or len(names) >= want:
                break
            url = BASE + nxt[0] if nxt[0].startswith("/") else nxt[0]
    finally:
        await page.close()
    return list(dict.fromkeys(names))[:want]


async def collect_negatives(ctx):
    page = await ctx.new_page()
    try:
        await page.goto(f"{BASE}/Category:按姓氏分类", timeout=60000, wait_until="domcontentloaded")
        await page.wait_for_timeout(4000)
        hrefs = await page.eval_on_selector_all("#mw-subcategories a", "els => els.map(e => e.getAttribute('href'))")
    finally:
        await page.close()
    hrefs = [h for h in hrefs if h and "/Category:" in h][:200]
    sem = asyncio.Semaphore(CONCURRENCY)
    names = []

    async def one(href):
        async with sem:
            pg = await ctx.new_page()
            try:
                await pg.goto(BASE + href if href.startswith("/") else href, timeout=60000, wait_until="domcontentloaded")
                await pg.wait_for_timeout(3000)
                got = await pg.eval_on_selector_all("#mw-pages a", "els => els.map(e => e.textContent.trim())")
                for x in got:
                    c = clean_title(x)
                    if valid(c):
                        names.append(c)
            except Exception:
                pass
            finally:
                await pg.close()

    await asyncio.gather(*[one(h) for h in hrefs])
    return list(dict.fromkeys(names))


def write_jsonl(path, rows):
    seen = set()
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            key = (r["name"], r["label"])
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved", path, len(rows))


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, executable_path=EDGE)
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36")
        pos_rows = []
        for label, cat in POS_CATS.items():
            names = await collect_category(ctx, cat)
            print(label, cat, len(names), names[:5])
            pos_rows += [{"name": n, "label": label} for n in names]
        neg = await collect_negatives(ctx)
        await browser.close()
    pos_names = {r["name"] for r in pos_rows}
    neg_rows = [{"name": n, "label": "normal"} for n in neg if n not in pos_names]
    write_jsonl(DATA / "moe_expression.jsonl", pos_rows)
    write_jsonl(DATA / "moe_negatives.jsonl", neg_rows)


if __name__ == "__main__":
    asyncio.run(main())

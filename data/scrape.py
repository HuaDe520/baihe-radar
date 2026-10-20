#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百合雷达 · 数据抓取脚本（晋江文学城作品库 → 三类主角名数据集）

来源
----
列表页：https://www.jjwxc.net/bookbase.php?fw0={fw0}&fbsj={fbsj}&ycx0={ycx0}&xx0={xx0}
        &mainview0={mainview0}&sd0={sd0}&lx0={lx0}&sortType={sortType}&isfinish={isfinish}
        &collectiontypes=ors&searchkeywords=&page={page}
    xx0=1 → 言情(romance)；xx0=2 → 纯爱/BL(purelove)；xx0=3 → 百合(baihe)
    未登录时每个筛选视图最多 10 页（第 11 页返回「请登入后再访问此页面」），
    因此叠加多个低重叠的筛选/排序视图（主视图：已完结/连载中；扩量视图：积分/字数排序、
    免费库、阅文衍生、衍生作品、发表时间、时代、视角等）拼够书目。
详情页：https://www.jjwxc.net/onebook.php?novelid={id}
    从 <meta name="Keywords"> 的「主角：」字段提取主角名；
    用 <span itemprop="genre"> 校验文章类型（百合 / 纯爱 / 言情）防止串类。

用法
----
  python data/scrape.py                     # 完整抓取（主视图 + 扩量视图，每类唯一名目标 6400）
  python data/scrape.py --pages 2 --limit 400   # 快速试跑
  python data/scrape.py --rebuild            # 不联网，仅用 data/_books.jsonl 重建交付文件
  python data/scrape.py --reset              # 清空中间文件后重抓
  python data/scrape.py --no-resume          # 忽略断点，重抓已完成的书

产物（均在 data/ 下）
----
  names.jsonl        全量原始抽取（一行一 JSON：name/label/genre/novelid，未去重）
  names_clean.jsonl  去重、剔除跨类歧义名后的训练集
  stats.json         计数统计
  DATA_REPORT.md     方法与结果报告
  progress.json      断点续抓进度（已处理 novelid）
  _books.jsonl       中间文件：每本书一行（novelid/label/genre/names），断点续抓依据
  _failures.jsonl    中间文件：解析失败/被丢弃的书（novelid/label/status）
  _list_meta.json    中间文件：列表页抓取元信息（页数、候选书目、各视图 novelid）
  names_v1.jsonl / names_clean_v1.jsonl / stats_v1.json  扩量前的 v1 备份（只读对比用）

只用标准库（urllib / gzip / zlib / re / json / concurrent.futures ...），Python 3.10+。
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import random
import re
import ssl
import sys
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------- 路径与常量

DATA_DIR = Path(__file__).resolve().parent
BOOKS_FILE = DATA_DIR / "_books.jsonl"          # 中间：成功解析的书
FAIL_FILE = DATA_DIR / "_failures.jsonl"        # 中间：丢弃/失败的书
LIST_META_FILE = DATA_DIR / "_list_meta.json"   # 中间：列表页元信息
RUN_META_FILE = DATA_DIR / "_run_meta.json"     # 中间：累计网络/耗时统计
PROGRESS_FILE = DATA_DIR / "progress.json"      # 断点续抓进度
NAMES_FILE = DATA_DIR / "names.jsonl"           # 交付：全量原始
CLEAN_FILE = DATA_DIR / "names_clean.jsonl"     # 交付：训练集
STATS_FILE = DATA_DIR / "stats.json"            # 交付：统计
REPORT_FILE = DATA_DIR / "DATA_REPORT.md"       # 交付：报告

LABELS = [("baihe", 3, "百合"), ("purelove", 2, "纯爱(BL)"), ("romance", 1, "言情")]
LABEL_ORDER = [name for name, _xx, _zh in LABELS]
LABEL_XX = {name: xx for name, xx, _zh in LABELS}
LABEL_ZH = {name: zh for name, _xx, zh in LABELS}
LABEL_KEYWORD = {"baihe": "百合", "purelove": "纯爱", "romance": "言情"}

# 站点未登录时每个筛选视图最多 10 页（第 11 页要求登录）。做成常量可调。
PAGES_PER_VIEW = 10
# 主视图：已完结 / 连载中 —— 两者书目几乎不重叠，各 10 页即每频道 20 个列表页。
VIEWS = [
    dict(isfinish=1, sortType=0, name="已完结"),
    dict(isfinish=2, sortType=0, name="连载中"),
]
# 扩量视图：主视图之外叠加不同排序 / 范围 / 原创性 / 发表时间 / 时代 / 视角筛选，
# 每个视图独立享受未登录 10 页上限，可拿到与主视图不重叠的书目。
# 顺序按「新增书目密度」从高到低：排序类 → 范围/衍生 → 时间 → 排序×完结状态 → 时代/视角。
EXPANSION_VIEWS = [
    dict(sortType=1, name="积分排序"),
    dict(sortType=3, name="字数排序"),
    dict(fw0=4, name="免费库"),
    dict(fw0=5, name="阅文衍生"),
    dict(ycx0=2, name="衍生作品"),
    dict(fbsj=2, name="发表时间近三月"),
    dict(fbsj=3, name="发表时间近一年"),
    dict(sortType=5, name="评论排序"),
    dict(sortType=1, isfinish=1, name="积分+已完结"),
    dict(sortType=1, isfinish=2, name="积分+连载中"),
    dict(sortType=3, isfinish=1, name="字数+已完结"),
    dict(sortType=3, isfinish=2, name="字数+连载中"),
    dict(sd0=2, name="古色古香"),
    dict(sd0=1, name="近代现代"),
    dict(fw0=4, sortType=1, name="免费库+积分"),
    dict(fw0=4, sortType=3, name="免费库+字数"),
    dict(fw0=5, sortType=1, name="阅文衍生+积分"),
    dict(ycx0=2, sortType=1, name="衍生+积分"),
    dict(fbsj=2, sortType=1, name="近三月+积分"),
    dict(fbsj=2, sortType=3, name="近三月+字数"),
    dict(sd0=2, sortType=1, name="古色古香+积分"),
    dict(sd0=2, sortType=3, name="古色古香+字数"),
    dict(mainview0=3, name="主攻视角"),
    dict(mainview0=4, name="主受视角"),
]
TARGET_UNIQUE_PER_LABEL = 6400   # 每类「清洗后未跨类去歧」唯一名目标；留出歧义剔除余量

WORKERS = 4                      # 并发上限（契约：≤4）
DELAY_MIN, DELAY_MAX = 0.15, 0.40
RETRIES = 3
TIMEOUT = 25
MAX_SECONDS = 2400               # 总时长软上限（40 分钟），到点收工出结果

# 列表页参数默认值（view 字典可覆盖任意一项）；xx0 / page 单独传入。
LIST_DEFAULTS = dict(fw0=0, fbsj=0, ycx0=0, mainview0=0, sd0=0, lx0=0,
                     sortType=0, isfinish=0, collectiontypes="ors")
LIST_URL = ("https://www.jjwxc.net/bookbase.php?fw0={fw0}&fbsj={fbsj}&ycx0={ycx0}"
            "&xx0={xx0}&mainview0={mainview0}&sd0={sd0}&lx0={lx0}&sortType={sortType}"
            "&isfinish={isfinish}&collectiontypes={collectiontypes}"
            "&searchkeywords=&page={page}")
DETAIL_URL = "https://www.jjwxc.net/onebook.php?novelid={novelid}"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "close",
}

# ---------------------------------------------------------------- 正则与清洗规则

LIST_ID_RE = re.compile(r"onebook\.php\?novelid=(\d+)")
KEYWORDS_RE = re.compile(r'<meta\s+name="Keywords"\s+content="(.*?)"', re.S | re.I)
GENRE_RE = re.compile(r'itemprop="genre"[^>]*>(.*?)</span>', re.S | re.I)
PROTAGONIST_RE = re.compile(r"主角[：:]\s*(.*?)(?:┃|配角[：:]|其它[：:]|\|最新更新|$)", re.S)

NAME_ALLOWED_RE = re.compile(r"^[\u4e00-\u9fa5·]{2,6}$")
NAME_SPLIT_RE = re.compile(r"[，,、/／;；|\s]+")
BRACKET_RE = re.compile(r"[【\[（(][^】\]）)]*[】\]）)]")

# 明显不是人名的词（角色/群体/占位/职业称呼等），直接丢弃。
NAME_BLACKLIST = {
    "暂无", "无", "群像", "主角", "配角", "其他", "其它", "动物们", "众人", "路人们",
    "主角团", "无cp", "待定", "系统", "女主", "男主", "女配", "男配", "双男主", "双女主",
    "全员", "大家", "群众", "路人", "路人甲", "路人乙", "路人丙", "甲乙丙", "同学", "老师",
    "作者", "读者", "观众", "网友们", "弹幕", "记者", "记者们", "吃瓜群众", "反派", "炮灰",
    "未知", "不详", "无名氏", "无名", "若干", "各种", "各种人", "一堆人", "一群", "一干人等",
    "男人", "女人", "男生", "女生", "少年", "少女", "孩子", "孩子们", "两人", "三人", "众",
    "闺蜜", "朋友", "家人", "父母", "爸妈", "哥哥", "姐姐", "弟弟", "妹妹", "室友", "同事",
    "老板", "管家", "侍女", "丫鬟", "掌柜", "医生", "护士", "警察", "律师", "店员", "学生",
    "校长", "班主任", "皇帝", "皇后", "公主", "王子", "小姐", "少爷", "大人", "姑娘",
    "公子", "先生", "女士", "将军", "王爷", "太子", "皇上", "太后", "师傅", "师父",
    "系统君", "天道", "世界", "命运", "作者君", "小天使", "男嘉宾", "女嘉宾", "幼崽",
    "替身", "马甲", "一个", "其他角色", "各种角色", "众配角", "配角们", "妖怪", "妖兽",
    "仙人", "仙子", "魔王", "恶魔", "天使", "龙套", "群演", "工具人", "纸片人",
    "老公", "老婆", "男友", "女友", "未婚妻", "未婚夫", "甲乙丙丁", "灵根", "灵石",
    "灵兽", "灵器", "法器", "丹药", "小孩儿", "小孩子", "小朋友", "汉王", "汉王妃",
    "小狐妖", "真少爷", "一些人", "笑死人", "圈外人", "圈内人", "植物人", "动物人",
    "坏女人", "大明星", "大道", "大恶龙", "大美女", "大鬼", "百鬼", "魔鬼", "前任",
    "男朋友", "女性", "女主角", "某某", "玩家", "觉醒者",
}
# 4 字以上描述性短语常见的后缀（「韦恩老爷」「安乐公主」这类称呼）
JUNK_SUFFIXES = ("老爷", "大人", "先生", "小姐", "少爷", "姑娘", "公子", "夫人",
                 "公主", "王子", "殿下", "陛下", "娘娘", "太后", "皇帝", "皇后",
                 "总裁", "老板", "医生", "老师", "同学", "少年", "少女", "等人",
                 "众人", "们", "王", "妃", "尊", "妖", "魔", "帝", "后", "人", "者")
# 名字里几乎不会出现的功能字（防止「什么的」「他们」「一个酷哥」之类混入）
NAME_BANNED_CHARS = set("的了着过吗呢吧啊呀哦嘛们众个替攻受没泥")

# 4 字及以上、且不含间隔号的词，只有「以姓氏开头」或「含音译常用字」才当人名。
# 用于过滤「一个酷哥 / 替身使者 / 神秘男嘉宾 / 混血幼崽」这类描述性短语，
# 同时保住「玛丽安娜 / 拜尔诺玛 / 尤利西斯 / 欧阳娜娜」这类名字。
SURNAME_START = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘"
    "葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞"
    "齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜"
    "阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘缪"
    "干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚程嵇邢滑裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜"
    "松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸"
    "司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴郁胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉"
    "宰郦雍却璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终"
    "暨居衡步都耿满弘匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙"
    "乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公拜桃上官司徒司空"
)
COMPOUND_SURNAMES = (
    "欧阳", "太史", "端木", "上官", "司马", "东方", "独孤", "南宫", "万俟", "闻人", "夏侯",
    "诸葛", "尉迟", "公羊", "赫连", "澹台", "皇甫", "宗政", "濮阳", "公冶", "太叔", "申屠",
    "公孙", "慕容", "仲孙", "钟离", "长孙", "宇文", "司徒", "鲜于", "司空", "闾丘", "子车",
    "亓官", "司寇", "巫马", "公西", "颛孙", "壤驷", "公良", "漆雕", "乐正", "宰父", "谷梁",
    "拓跋", "夹谷", "轩辕", "令狐", "段干", "百里", "呼延", "东郭", "南门", "羊舌", "微生",
    "公户", "公玉", "公仪", "梁丘", "公仲", "公上", "公门", "公山", "公坚", "左丘", "公伯",
    "西门", "公祖", "第五", "公乘", "南荣", "东里", "东宫", "仲长", "子书", "子桑", "即墨",
    "达奚", "褚师", "慕容",
)
TRANSLIT_CHARS = set("玛娜丽莎莉蒂娅芙蕾丝黛琳露妮尔斯克姆拉诺维卡洛恩森伊亚德安贝米泽瓦托塞普奥"
                     "休戴舍乔希珍·")


def _filter_tokens(tokens: list[str]) -> list[str]:
    """对已切分 token 应用全部清洗规则（供抓取时与 --rebuild 重清洗共用）。"""
    out: list[str] = []
    for tok in tokens:
        tok = tok.strip().strip("·、，。.")
        if not tok:
            continue
        if not NAME_ALLOWED_RE.match(tok):
            continue
        if tok.lower() in NAME_BLACKLIST or tok in NAME_BLACKLIST:
            continue
        if tok.endswith("等") or tok.endswith(JUNK_SUFFIXES):
            continue
        if any(ch in NAME_BANNED_CHARS for ch in tok):
            continue
        if len(tok) >= 5 and "·" not in tok:
            continue                  # 中文人名几乎不超过 4 字；5+ 多为人名粘连或整句
        if len(tok) == 4 and "·" not in tok:
            ok = (tok[0] in SURNAME_START
                  or any(tok.startswith(c) for c in COMPOUND_SURNAMES))
            if not ok:
                # 音译名（无姓氏开头）：至少两个字属于音译常用字，如「玛丽安娜」
                if sum(1 for ch in tok if ch in TRANSLIT_CHARS) < 2:
                    continue
        if tok in out:            # 同书内去重
            continue
        out.append(tok)
    return out


def clean_names(raw: str) -> tuple[list[str], int, list[str]]:
    """把「主角：」字段切成名字列表；返回 (合法名列表, 原始 token 数, 原始 token 列表)。"""
    raw = BRACKET_RE.sub(" ", raw or "")
    tokens = [t for t in NAME_SPLIT_RE.split(raw) if t]
    return _filter_tokens(tokens), len(tokens), tokens


# ---------------------------------------------------------------- 抓取器


class Fetcher:
    """带并发限速 / 重试退避 / gzip 增量解压 / 403 熔断的 HTTP 抓取器。"""

    def __init__(self, retries: int = RETRIES, timeout: int = TIMEOUT):
        self.retries = retries
        self.timeout = timeout
        self.ctx = ssl.create_default_context()
        self.requests = 0
        self.bytes_read = 0
        self.forbidden = 0
        self.retried = 0
        self.aborted = False
        self.abort_reason = ""

    # -- 基础：一次请求，带 gzip 增量解压 + 提前截断（meta 在 <head>，不用整页） --
    def _get_text(self, url: str, cap: int, probe) -> str:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=self.timeout, context=self.ctx) as resp:
            enc_hdr = (resp.headers.get("Content-Encoding") or "").lower()
            charset = (resp.headers.get_content_charset() or "").lower()
            gz = "gzip" in enc_hdr
            dec = zlib.decompressobj(16 + zlib.MAX_WBITS) if gz else None
            buf = bytearray()
            text = ""
            total = 0
            while total < cap:
                chunk = resp.read(16384)
                if not chunk:
                    break
                total += len(chunk)
                self.bytes_read += len(chunk)
                if dec is not None:
                    try:
                        buf += dec.decompress(chunk)
                    except zlib.error:
                        dec = None
                        buf += chunk
                else:
                    buf += chunk
                if not gz and buf[:2] == b"\x1f\x8b":     # 未声明但实际是 gzip
                    try:
                        buf = bytearray(zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(bytes(buf)))
                        gz = True
                    except zlib.error:
                        pass
                text = _decode(bytes(buf), charset)
                if probe is not None and probe(text):
                    break
            self.requests += 1
            return text

    def get(self, url: str, cap: int, probe=None) -> str:
        last_exc = None
        for attempt in range(1, self.retries + 1):
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            try:
                return self._get_text(url, cap, probe)
            except HTTPError as exc:
                last_exc = exc
                if exc.code == 403:
                    self.forbidden += 1
                    if self.forbidden >= 10:
                        self.aborted = True
                        self.abort_reason = "连续 403 / 反爬熔断（累计 10 次）"
                        raise
                elif exc.code in (404, 410):
                    raise
            except (URLError, TimeoutError, OSError, zlib.error) as exc:
                last_exc = exc
            if attempt < self.retries:
                self.retried += 1
                time.sleep(0.8 * attempt + random.uniform(0.0, 0.6))
        raise last_exc if last_exc else RuntimeError("request failed")


def _decode(raw: bytes, charset: str = "") -> str:
    order = []
    if charset in ("utf-8", "utf8"):
        order.append("utf-8")
    order.append("gb18030")
    if charset in ("gb2312", "gbk", "gb18030"):
        order.insert(0, "gb18030")
    order.append("utf-8")
    for enc in dict.fromkeys(order):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("gb18030", "ignore")


# ---------------------------------------------------------------- 列表页


def build_list_url(label: str, view: dict, page: int) -> str:
    params = dict(LIST_DEFAULTS)
    params.update({k: v for k, v in view.items() if k in LIST_DEFAULTS})
    params["xx0"] = LABEL_XX[label]
    params["page"] = page
    return LIST_URL.format(**params)


def view_key(view: dict) -> str:
    return "|".join(f"{k}={view[k]}" for k in sorted(view) if k in LIST_DEFAULTS)


def fetch_list_ids(fetcher: Fetcher, label: str, view: dict, pages: int) -> tuple[list[int], int]:
    """抓一个筛选视图的前 pages 页，返回 (去重后的 novelid 列表, 实际抓到的页数)。"""
    ids: list[int] = []
    seen: set[int] = set()
    pages_ok = 0
    for page in range(1, pages + 1):
        url = build_list_url(label, view, page)
        try:
            text = fetcher.get(url, cap=2 * 1024 * 1024)
        except Exception as exc:  # 列表页失败不致命，记录后继续下一页
            print(f"  [warn] 列表页失败 {label} {view['name']} p{page}: {exc!r}")
            continue
        page_ids = [int(x) for x in dict.fromkeys(LIST_ID_RE.findall(text))]
        if not page_ids:
            break                       # 页数耗尽 / 登录墙
        new = [i for i in page_ids if i not in seen]
        if not new:
            break                       # 排序漂移导致重复，提前停
        for i in new:
            seen.add(i)
            ids.append(i)
        pages_ok += 1
        print(f"  [{label}] 列表 p{page}（{view['name']}）-> {len(page_ids)} 本，新增 {len(new)}，累计 {len(ids)}")
        if len(page_ids) < 50:          # 尾部不足一页，说明该视图已到底
            break
    return ids, pages_ok


def collect_list_phase(fetcher: Fetcher, pages: int, views: int) -> dict:
    """抓主视图列表页，返回 list_meta；已抓过的视图结果可复用（不重复抓）。"""
    list_meta: dict = {}
    for label, _xx, _zh in LABELS:
        view_metas = []
        for view in VIEWS[:views]:
            ids, pages_ok = fetch_list_ids(fetcher, label, view, pages)
            view_metas.append({**view, "key": view_key(view), "pages": pages_ok, "ids": ids})
        list_meta[label] = {"views": view_metas, "pages": sum(v["pages"] for v in view_metas)}
        print(f"[{label}] 主视图列表完成：{list_meta[label]['pages']} 页，候选书目 "
              f"{len({i for v in view_metas for i in v['ids']})} 本")
    return list_meta


def append_extra_view(fetcher: Fetcher, list_meta: dict, label: str, view: dict, pages: int) -> list[int]:
    ids, pages_ok = fetch_list_ids(fetcher, label, view, pages)
    list_meta[label]["views"].append({**view, "key": view_key(view),
                                      "pages": pages_ok, "ids": ids})
    list_meta[label]["pages"] += pages_ok
    return ids


# ---------------------------------------------------------------- 详情页解析


def parse_detail(text: str, label: str) -> dict:
    """解析详情页，返回 {status, genre, names, raw_tokens}。"""
    low = text
    if "请 登入 后再访问" in low or "请登入后再访问" in low:
        return {"status": "login_wall", "genre": "", "names": [], "raw_tokens": 0}
    kw = KEYWORDS_RE.search(text)
    if not kw:
        return {"status": "no_keywords", "genre": "", "names": [], "raw_tokens": 0}
    genre_m = GENRE_RE.search(text)
    genre = ""
    if genre_m:
        genre = html_mod.unescape(re.sub(r"<[^>]+>", "", genre_m.group(1))).strip()
    if LABEL_KEYWORD[label] not in genre:
        return {"status": "genre_mismatch", "genre": genre, "names": [], "raw_tokens": 0}
    kw_text = html_mod.unescape(kw.group(1))
    m = PROTAGONIST_RE.search(kw_text)
    if not m:
        return {"status": "no_protagonist", "genre": genre, "names": [], "raw_tokens": 0,
                "keywords": kw_text[:200]}
    names, raw_n, raw_tokens = clean_names(m.group(1))
    return {"status": "ok", "genre": genre, "names": names, "raw_tokens": raw_n,
            "raw": raw_tokens[:12]}


def fetch_and_parse(fetcher: Fetcher, novelid: int, label: str) -> dict:
    url = DETAIL_URL.format(novelid=novelid)
    text = fetcher.get(url, cap=160 * 1024, probe=KEYWORDS_RE.search)
    res = parse_detail(text, label)
    res["novelid"] = novelid
    res["label"] = label
    return res


# ---------------------------------------------------------------- 主抓取循环


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue        # 容忍中断产生的半行
    return rows


def _rewrite_jsonl(path: Path, rows) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_done() -> tuple[set[int], dict[str, dict[int, dict]], dict[str, int]]:
    """从中间文件恢复：已完成 novelid、每本书记录、失败计数。

    顺带按当前清洗规则重清洗历史记录的原始 token（``raw`` 字段）；
    旧格式（无 raw）无法重清洗，丢弃后会在本次重抓。这样改了规则重跑即可升级数据。
    """
    done: set[int] = set()
    books: dict[int, dict] = {}
    fail_counts: dict[str, int] = {}
    books_changed = False
    for row in read_jsonl(BOOKS_FILE):
        nid = row.get("novelid")
        if not isinstance(nid, int) or "raw" not in row:
            books_changed = True
            continue
        names = _filter_tokens(row.get("raw") or [])
        if names != row.get("names"):
            books_changed = True
            row["names"] = names
        done.add(nid)
        books[nid] = row
    if books_changed:
        _rewrite_jsonl(BOOKS_FILE, (books[nid] for nid in sorted(books)))
    for row in read_jsonl(FAIL_FILE):
        nid = row.get("novelid")
        if isinstance(nid, int):
            done.add(nid)
        status = row.get("status", "unknown")
        fail_counts[status] = fail_counts.get(status, 0) + 1
    return done, books, fail_counts


def save_progress(done: set[int], books: dict[int, dict], t0: float) -> None:
    payload = {
        "updated_at": now_iso(),
        "done_count": len(done),
        "done_novelids": sorted(done),
        "books_ok": len(books),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    tmp = PROGRESS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(PROGRESS_FILE)


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_run_meta() -> dict:
    """累计运行元数据（多轮抓取/续跑的网络量与耗时），用于报告里的「累计」统计。"""
    empty = {"runs": 0, "requests": 0, "detail_requests": 0, "bytes_read": 0,
             "retries": 0, "forbidden": 0, "elapsed_sec": 0.0,
             "first_run_at": None, "last_run_at": None, "history_note": ""}
    if RUN_META_FILE.exists():
        try:
            meta = json.loads(RUN_META_FILE.read_text(encoding="utf-8"))
            empty.update({k: v for k, v in meta.items() if k in empty})
        except (json.JSONDecodeError, OSError):
            pass
    return empty


def update_run_meta(prev: dict, fetcher_stats: dict, elapsed: float) -> dict:
    meta = dict(prev)
    meta["runs"] = int(prev.get("runs", 0)) + 1
    meta["requests"] = int(prev.get("requests", 0)) + int(fetcher_stats.get("requests", 0))
    meta["detail_requests"] = int(prev.get("detail_requests", 0)) + int(fetcher_stats.get("detail_requests", 0))
    meta["bytes_read"] = int(prev.get("bytes_read", 0)) + int(fetcher_stats.get("bytes_read", 0))
    meta["retries"] = int(prev.get("retries", 0)) + int(fetcher_stats.get("retried", 0))
    meta["forbidden"] = int(prev.get("forbidden", 0)) + int(fetcher_stats.get("forbidden", 0))
    meta["elapsed_sec"] = round(float(prev.get("elapsed_sec", 0.0)) + elapsed, 1)
    meta["last_run_at"] = now_iso()
    meta["first_run_at"] = prev.get("first_run_at") or meta["last_run_at"]
    RUN_META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def process_details(fetcher: Fetcher, queues: dict[str, list[int]], done: set[int],
                    books: dict[int, dict], fail_counts: dict[str, int],
                    label_unique: dict[str, set[str]], args, t0: float) -> None:
    """并发抓详情页；每类唯一名达到 target 即停该类。结果由主线程单线程落盘。"""
    label_ok = {lab: 0 for lab in LABEL_ORDER}
    label_done_this_run = {lab: 0 for lab in LABEL_ORDER}
    cursor = {lab: 0 for lab in LABEL_ORDER}
    deadline = t0 + args.max_seconds
    books_fh = BOOKS_FILE.open("a", encoding="utf-8", buffering=1)
    fail_fh = FAIL_FILE.open("a", encoding="utf-8", buffering=1)
    total_requests0 = fetcher.requests
    last_progress = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending: dict = {}

            def next_task():
                for _ in range(len(LABEL_ORDER) * 2):
                    for lab in LABEL_ORDER:
                        if label_unique.get(lab) and len(label_unique[lab]) >= args.target:
                            continue
                        if label_done_this_run[lab] >= args.limit > 0:
                            continue
                        q = queues.get(lab, [])
                        while cursor[lab] < len(q) and (q[cursor[lab]] in done
                                                        or q[cursor[lab]] in pending_nids):
                            cursor[lab] += 1
                        if cursor[lab] < len(q):
                            nid = q[cursor[lab]]
                            cursor[lab] += 1
                            return lab, nid
                return None

            pending_nids: set[int] = set()
            while True:
                if fetcher.aborted:
                    print(f"  [abort] {fetcher.abort_reason}")
                    break
                if time.time() > deadline:
                    print("  [abort] 达到总时长软上限，停止抓取，用已有数据出结果")
                    break
                while len(pending) < args.workers * 2:
                    task = next_task()
                    if task is None:
                        break
                    lab, nid = task
                    pending_nids.add(nid)
                    fut = pool.submit(fetch_and_parse, fetcher, nid, lab)
                    pending[fut] = (lab, nid)
                if not pending:
                    break
                done_futs, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=5)
                for fut in done_futs:
                    lab, nid = pending.pop(fut)
                    pending_nids.discard(nid)
                    try:
                        res = fut.result()
                    except Exception as exc:   # 网络反复失败：不标记完成，下次重跑再试
                        fail_counts["network_error"] = fail_counts.get("network_error", 0) + 1
                        print(f"  [warn] {lab} {nid} 网络失败: {exc!r}")
                        continue
                    status = res["status"]
                    if status == "ok":
                        row = {"novelid": nid, "label": lab, "genre": res["genre"],
                               "names": res["names"], "raw_tokens": res["raw_tokens"],
                               "raw": res.get("raw", [])}
                        books[nid] = row
                        done.add(nid)
                        label_ok[lab] += 1
                        label_done_this_run[lab] += 1
                        label_unique.setdefault(lab, set()).update(res["names"])
                        books_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    else:
                        fail_counts[status] = fail_counts.get(status, 0) + 1
                        done.add(nid)
                        fail_fh.write(json.dumps({"novelid": nid, "label": lab, "status": status,
                                                  "genre": res.get("genre", "")},
                                                 ensure_ascii=False) + "\n")
                        if status == "genre_mismatch":
                            label_done_this_run[lab] += 1
                if time.time() - last_progress > 30:
                    save_progress(done, books, t0)
                    last_progress = time.time()
                    el = int(time.time() - t0)
                    print(f"  进度 {el}s：详情请求 {fetcher.requests - total_requests0} 次，"
                          f"成功 {sum(label_ok.values())} 本；唯一名 "
                          + ", ".join(f"{lab}={len(label_unique.get(lab, ()))}" for lab in LABEL_ORDER))
    finally:
        books_fh.close()
        fail_fh.close()
        save_progress(done, books, t0)


# ---------------------------------------------------------------- 产物构建


def build_outputs(list_meta: dict, books: dict[int, dict], fail_counts: dict[str, int],
                  fetcher_stats: dict, elapsed: float, args, run_meta: dict | None = None) -> dict:
    """从每本书记录重建 names.jsonl / names_clean.jsonl / stats.json / DATA_REPORT.md。"""
    run_meta = run_meta or {}
    rows = [books[nid] for nid in sorted(books)]
    by_label_books: dict[str, list[dict]] = {lab: [] for lab in LABEL_ORDER}
    for row in rows:
        by_label_books.setdefault(row["label"], []).append(row)

    # --- names.jsonl：全量原始（同一名字在不同书中多次出现则保留多行） ---
    raw_rows: list[dict] = []
    for lab in LABEL_ORDER:
        for row in by_label_books[lab]:
            for name in row["names"]:
                raw_rows.append({"name": name, "label": lab,
                                 "genre": row["genre"], "novelid": row["novelid"]})
    with NAMES_FILE.open("w", encoding="utf-8") as fh:
        for r in raw_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- 跨类歧义：同一名字出现在多个 label，全部剔除 ---
    labels_by_name: dict[str, set[str]] = {}
    for r in raw_rows:
        labels_by_name.setdefault(r["name"], set()).add(r["label"])
    ambiguous = {n for n, labs in labels_by_name.items() if len(labs) > 1}
    ambiguous_by_label_pair: dict[str, int] = {}
    for n in ambiguous:
        key = "+".join(sorted(labels_by_name[n]))
        ambiguous_by_label_pair[key] = ambiguous_by_label_pair.get(key, 0) + 1

    # --- names_clean.jsonl：类内去重 + 剔除跨类歧义 ---
    clean_rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    raw_unique_by_label: dict[str, set[str]] = {lab: set() for lab in LABEL_ORDER}
    for r in raw_rows:
        raw_unique_by_label[r["label"]].add(r["name"])
        if r["name"] in ambiguous:
            continue
        key = (r["label"], r["name"])
        if key in seen:
            continue
        seen.add(key)
        clean_rows.append(dict(r))
    with CLEAN_FILE.open("w", encoding="utf-8") as fh:
        for r in clean_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_label = {lab: 0 for lab in LABEL_ORDER}
    clean_by_label: dict[str, list[dict]] = {lab: [] for lab in LABEL_ORDER}
    for r in clean_rows:
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1
        clean_by_label[r["label"]].append(r)

    # --- stats.json ---
    books_by_label = {lab: len(by_label_books[lab]) for lab in LABEL_ORDER}
    raw_occ = {lab: 0 for lab in LABEL_ORDER}
    for r in raw_rows:
        raw_occ[r["label"]] += 1
    pages_per_label = {lab: list_meta.get(lab, {}).get("pages", 0) for lab in LABEL_ORDER}
    views_info = {}
    for lab in LABEL_ORDER:
        views_info[lab] = [
            {k: v[k] for k in ("name", "isfinish", "sortType", "fw0", "ycx0", "fbsj",
                               "sd0", "mainview0", "pages") if k in v}
            | {"ids": len(v["ids"])}
            for v in list_meta.get(lab, {}).get("views", [])
        ]
    candidate_books = {
        lab: len({nid for v in list_meta.get(lab, {}).get("views", []) for nid in v["ids"]})
        for lab in LABEL_ORDER
    }
    # 与 v1 基线对比（若交付目录中存在 names_v1 / stats_v1 备份）
    v1_baseline = None
    v1_stats = DATA_DIR / "stats_v1.json"
    if v1_stats.exists():
        try:
            old = json.loads(v1_stats.read_text(encoding="utf-8"))
            v1_baseline = {"generated_at": old.get("generated_at"), "by_label": old.get("by_label"),
                           "books": old.get("books"),
                           "ambiguous_names": old.get("ambiguous_names")}
        except (json.JSONDecodeError, OSError):
            v1_baseline = None
    # 失败计数需要区分 label：从 _failures.jsonl 重新聚一次
    fail_by_label: dict[str, dict[str, int]] = {lab: {} for lab in LABEL_ORDER}
    fail_total: dict[str, int] = {}
    for row in read_jsonl(FAIL_FILE):
        lab = row.get("label")
        st = row.get("status", "unknown")
        fail_total[st] = fail_total.get(st, 0) + 1
        if lab in fail_by_label:
            fail_by_label[lab][st] = fail_by_label[lab].get(st, 0) + 1
    attempted = {}
    for lab in LABEL_ORDER:
        attempted[lab] = books_by_label[lab] + sum(fail_by_label[lab].values())
    raw_token_total = sum(int(r.get("raw_tokens", 0) or 0) for r in rows)
    accepted_token_total = sum(len(r["names"]) for r in rows)

    stats = {
        "generated_at": now_iso(),
        "history_note": run_meta.get("history_note", ""),
        "source": "晋江文学城作品库 https://www.jjwxc.net/bookbase.php (gb18030/gzip)",
        "detail_url_template": DETAIL_URL,
        "labels": {lab: {"xx0": LABEL_XX[lab], "zh": LABEL_ZH[lab],
                         "genre_keyword": LABEL_KEYWORD[lab]} for lab in LABEL_ORDER},
        "by_label": by_label,
        "books": books_by_label,
        "ambiguous_names": len(ambiguous),
        "pages_per_label": pages_per_label,
        "views_per_label": views_info,
        "candidate_books": candidate_books,
        "books_attempted": attempted,
        "books_discarded": {lab: sum(fail_by_label[lab].values()) for lab in LABEL_ORDER},
        "discard_reasons": fail_total,
        "raw_name_occurrences": raw_occ,
        "raw_unique_names": {lab: len(raw_unique_by_label[lab]) for lab in LABEL_ORDER},
        "protagonist_tokens": {"total": raw_token_total, "accepted": accepted_token_total,
                               "rejected": raw_token_total - accepted_token_total},
        "ambiguous_by_label_pair": ambiguous_by_label_pair,
        "ambiguous_examples": sorted(ambiguous)[:30],
        "v1_baseline": v1_baseline,
        "added_since_v1": (
            {"by_label": {lab: by_label[lab] - v1_baseline["by_label"][lab] for lab in LABEL_ORDER},
             "books": {lab: books_by_label[lab] - v1_baseline["books"][lab] for lab in LABEL_ORDER}}
            if v1_baseline else None),
        "expansion_views_used": getattr(args, "expansion_used", []),
        "network": {
            "this_run_requests": fetcher_stats.get("requests", 0),
            "this_run_detail_requests": fetcher_stats.get("detail_requests", 0),
            "this_run_elapsed_sec": round(elapsed, 1),
            "cumulative_requests": run_meta.get("requests", 0),
            "cumulative_bytes_read": run_meta.get("bytes_read", 0),
            "cumulative_retries": run_meta.get("retries", 0),
            "forbidden_403": run_meta.get("forbidden", 0),
            "crawl_runs": run_meta.get("runs", 0),
            "cumulative_elapsed_sec": run_meta.get("elapsed_sec", 0.0),
            "aborted": fetcher_stats.get("aborted", False),
            "abort_reason": fetcher_stats.get("abort_reason", ""),
        },
        "params": {
            "pages_per_view": args.pages, "views": args.views, "target_per_label": args.target,
            "workers": args.workers, "retries": RETRIES, "timeout_sec": TIMEOUT,
            "delay": [DELAY_MIN, DELAY_MAX], "max_seconds": args.max_seconds,
        },
        "names_jsonl_rows": len(raw_rows),
        "names_clean_jsonl_rows": len(clean_rows),
        "elapsed_sec_this_run": round(elapsed, 1),
    }
    STATS_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # --- DATA_REPORT.md ---
    write_report(stats, clean_by_label, raw_rows, args)
    return stats


def write_report(stats: dict, clean_by_label: dict[str, list[dict]], raw_rows: list[dict],
                 args) -> None:
    rng = random.Random(20261020)
    sample_blocks = []
    for lab in LABEL_ORDER:
        names = [r["name"] for r in clean_by_label[lab]]
        sample = rng.sample(names, min(20, len(names))) if names else []
        sample_blocks.append(f"**{LABEL_ZH[lab]}（{lab}）**：{'、'.join(sample)}")
    samples_md = "\n\n".join(sample_blocks)

    L = []
    L.append("# 「百合雷达」数据抓取报告（DATA_REPORT）\n")
    L.append(f"- 生成时间：{stats['generated_at']}")
    L.append(f"- 数据源：晋江文学城作品库（jjwxc.net），公开页面，仅抓取虚构小说作品的"
             f"「主角」字段，用于娱乐向玩具模型训练。")
    L.append(f"- 交付文件：`data/names.jsonl`（全量原始 {stats['names_jsonl_rows']} 行）、"
             f"`data/names_clean.jsonl`（训练集 {stats['names_clean_jsonl_rows']} 行）、"
             f"`data/stats.json`、`data/scrape.py`；"
             f"扩量前的 v1 备份：`data/names_v1.jsonl`、`data/names_clean_v1.jsonl`、`data/stats_v1.json`。")
    L.append("")
    L.append("## 一、结论速览\n")
    L.append("| 频道 | label | 抓取书数 | 原始名字出现次数 | 清洗后唯一名 | 最终训练名数 | 列表页数 |")
    L.append("|---|---|---|---|---|---|---|")
    for lab in LABEL_ORDER:
        L.append(f"| {LABEL_ZH[lab]} | {lab} | {stats['books'][lab]} | {stats['raw_name_occurrences'][lab]} "
                 f"| {stats['raw_unique_names'][lab]} | {stats['by_label'][lab]} | {stats['pages_per_label'][lab]} |")
    L.append("")
    L.append(f"- 跨类歧义名（同名出现在多个频道）：**{stats['ambiguous_names']}** 个，已全部从训练集剔除。")
    L.append(f"- 总计训练样本：**{sum(stats['by_label'].values())}** 条（每类 "
             + " / ".join(f"{LABEL_ZH[lab]} {stats['by_label'][lab]}" for lab in LABEL_ORDER) + "）。")
    v1 = stats.get("v1_baseline")
    if v1 and stats.get("added_since_v1"):
        add = stats["added_since_v1"]
        L.append(f"- **扩量对比**：v1（{v1.get('generated_at', '')[:19]}）每类 "
                 + " / ".join(f"{LABEL_ZH[lab]} {v1['by_label'].get(lab, 0)}" for lab in LABEL_ORDER)
                 + f"（合计 {sum(v1['by_label'].values())}），本次**新增每类 "
                 + " / ".join(f"{add['by_label'][lab]}" for lab in LABEL_ORDER)
                 + f"**（合计 +{sum(add['by_label'].values())}），新增书目 "
                 + " / ".join(f"{add['books'][lab]}" for lab in LABEL_ORDER) + " 本。")
    L.append("")
    L.append("## 二、URL 模板与频道映射\n")
    L.append("列表页（每页 100 本，`page` 从 1 开始）：\n")
    L.append("```text")
    L.append("https://www.jjwxc.net/bookbase.php?fw0={fw0}&fbsj={fbsj}&ycx0={ycx0}&xx0={xx0}"
             "&mainview0={mainview0}&sd0={sd0}&lx0={lx0}&sortType={sortType}&isfinish={isfinish}"
             "&collectiontypes=ors&searchkeywords=&page={page}")
    L.append("```\n")
    L.append("参数取值：`xx0` 频道（1=言情/2=纯爱/3=百合）；`fw0` 范围（0=全站、4=免费库、5=阅文衍生）；"
             "`fbsj` 发表时间（0=全部、2=近三月、3=近一年）；`ycx0` 原创性（0=全部、2=衍生）；"
             "`sd0` 时代（1=近代现代、2=古色古香、4=幻想未来）；`mainview0` 视角（3=主攻、4=主受）；"
             "`sortType` 排序（0=更新时间、1=积分、3=字数、5=评论）；`isfinish` 完结状态（0=全部、1=已完结、2=连载中）。\n")
    L.append("| 频道 | xx0 | 校验用的文章类型关键词 |")
    L.append("|---|---|---|")
    for lab in LABEL_ORDER:
        L.append(f"| {LABEL_ZH[lab]} | {LABEL_XX[lab]} | `{LABEL_KEYWORD[lab]}` |")
    L.append("")
    L.append("详情页：`https://www.jjwxc.net/onebook.php?novelid={novelid}`\n")
    L.append("## 三、抓取策略\n")
    L.append("- **未登录每视图最多 10 页**：列表页第 11 页起返回「请登入后再访问此页面」。"
             "因此每个频道叠加多个低重叠的筛选/排序视图，每个视图各抓 10 页："
             "主视图 2 个（已完结、连载中）+ 扩量视图（积分排序、字数排序、免费库、阅文衍生、"
             "衍生作品、近三月/近一年、积分/字数×完结状态、古色古香等，按新增书目密度自动追加）。"
             f"最终每频道 **{stats['pages_per_label'].get('baihe', 0)} 个列表页**"
             f"（候选 {stats['candidate_books'].get('baihe', 0)} 本），"
             f"实际解析 {stats['books']['baihe']} / {stats['books']['purelove']} / {stats['books']['romance']} 本；"
             f"每类清洗后唯一名达到目标（{stats['params'].get('target_per_label', 0)}）即停止该类，省时省流量。")
    if stats.get("expansion_views_used"):
        used = "、".join(f"{e['name']}(+{e['new_books']})" for e in stats["expansion_views_used"])
        L.append(f"- 本次用到的扩量视图（括号内为本轮新增书目，三频道合计）：{used}。"
                 "完整视图/页数/候选书目明细见 `stats.json` 的 `views_per_label`。")
    L.append(f"- **反爬与礼貌**：并发 {args.workers}（≤4），每次请求前随机延时 "
             f"{DELAY_MIN}–{DELAY_MAX}s，失败指数退避重试 {RETRIES} 次，超时 {TIMEOUT}s，"
             "正常 Chrome UA + gzip；累计 403 达阈值自动熔断。")
    L.append("- **省带宽**：详情页只读到 `<head>` 中出现 Keywords 为止（通常 24–64KB 压缩流），"
             "不整页下载、不落盘原始 HTML。")
    L.append("- **断点续抓**：已完成/已判定丢弃的 novelid 记在 `progress.json`，重跑自动跳过；"
             "每 30 秒落盘一次，中断后直接重跑即可续抓。扩量视图的 novelid 与已处理集合去重，只抓新书。")
    L.append("- **去重与清洗**：同书内同名去重；类内同名去重；跨频道同名全部剔除（记入 stats）。")
    L.append("")
    L.append("## 四、名字抽取与清洗规则\n")
    L.append("1. 详情页 `<meta name=\"Keywords\">` → 正则 "
             "`主角[：:]\\s*(.*?)(?:┃|配角[：:]|其它[：:]|\\|最新更新|$)` 截取主角段；")
    L.append("2. 按 `[，,、/／;；|\\s]+` 切分，先去掉括号补充说明（如「（女主）」）；")
    L.append("3. 只保留 `^[\\u4e00-\\u9fa5·]{2,6}$`：2–6 个汉字、允许间隔号 `·`，"
             "不含数字/拉丁字母/标点；")
    L.append("4. 黑名单过滤明显非人名：人群/占位/职业称呼/系统等（`系统`、`主角`、`配角`、`众人`、"
             "`动物们`、`路人甲`、`暂无`、`待定`…），并剔除含「等/们」及功能字的词；")
    L.append("5. 4 字及以上且无间隔号的词，必须是「以姓氏开头」或「含音译常用字」才保留"
             "（过滤 `一个酷哥`、`替身使者`、`神秘男嘉宾`、`混血幼崽` 这类描述性短语，"
             "同时保 `玛丽安娜`、`尤利西斯`、`欧阳娜娜` 这类名字）；")
    L.append("6. `<span itemprop=\"genre\">` 校验文章类型含对应关键词，否则丢弃该书（防串类）。")
    L.append("")
    tk = stats.get("protagonist_tokens", {})
    L.append(f"主角字段共切出原始 token {tk.get('total', 0)} 个，清洗后保留 {tk.get('accepted', 0)} 个，"
             f"剔除 {tk.get('rejected', 0)} 个（非人名/格式不符）。")
    L.append("")
    L.append("## 五、每类随机 20 个样例名\n")
    L.append(samples_md + "\n")
    L.append("## 六、异常与处理\n")
    reasons = stats.get("discard_reasons", {})
    L.append("| 情况 | 处理 | 数量 |")
    L.append("|---|---|---|")
    L.append(f"| 列表页第 11 页起需登录 | 每个频道叠加多个低重叠筛选/排序视图，每视图各 10 页 | — |")
    L.append(f"| 详情页无 Keywords（作品不存在/下架） | 丢弃并记 `_failures.jsonl` | {reasons.get('no_keywords', 0)} |")
    L.append(f"| 文章类型不含本频道关键词（串类） | 丢弃该书记录 | {reasons.get('genre_mismatch', 0)} |")
    L.append(f"| Keywords 无「主角：」字段 | 丢弃该书记录 | {reasons.get('no_protagonist', 0)} |")
    L.append(f"| 命中登录墙页面 | 丢弃并记录 | {reasons.get('login_wall', 0)} |")
    L.append(f"| 网络超时/连接错误（重试 3 次后仍失败） | 不标记完成，下次重跑自动重试 | {reasons.get('network_error', 0)} |")
    L.append("")
    net = stats["network"]
    if net.get("this_run_requests"):
        L.append(f"- 网络统计（本次运行）：请求 {net.get('this_run_requests', 0)} 次，"
                 f"耗时 {net.get('this_run_elapsed_sec', 0) / 60:.1f} 分钟。")
    else:
        L.append("- 网络统计：本次为离线重建（`--rebuild`），未发起网络请求。")
    if net.get("crawl_runs"):
        L.append(f"- 网络统计（累计 {net.get('crawl_runs', 0)} 轮，自元数据启用起）："
                 f"请求 {net.get('cumulative_requests', 0)} 次，"
                 f"读取约 {net.get('cumulative_bytes_read', 0) / 1048576:.1f} MB，"
                 f"重试 {net.get('cumulative_retries', 0)} 次，403 次数 {net.get('forbidden_403', 0)}，"
                 f"累计耗时约 {net.get('cumulative_elapsed_sec', 0) / 60:.1f} 分钟。")
    if net.get("aborted"):
        L.append(f"- ⚠ 本次运行触发熔断：{net.get('abort_reason', '')}（用已抓数据出结果）。")
    L.append("- 注：本数据集经多轮抓取完成（试跑校准 → 首次全量 → 按最终清洗规则重建/补量），"
             "网络「累计」统计从启用运行元数据（`_run_meta.json`）后开始记录；"
             "成功解析书数以 `_books.jsonl` 为准。")
    if stats.get("history_note"):
        L.append(f"- 抓取历史（补记）：{stats['history_note']}")
    L.append("")
    L.append("## 七、复现方式\n")
    L.append("```bash")
    L.append("python data/scrape.py            # 全量抓取（断点续抓，可反复运行）")
    L.append("python data/scrape.py --rebuild  # 不联网，从 data/_books.jsonl 重建交付文件")
    L.append("python data/scrape.py --reset    # 清空中间文件重抓")
    L.append("```")
    L.append("")
    L.append("## 八、数据使用声明\n")
    L.append("训练数据全部来自虚构小说作品的「主角」字段，只用于娱乐向的玩具分类模型，"
             "不代表、也无法推断任何真实个人的性取向；网站必须显著位置展示该声明。")
    REPORT_FILE.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 入口


def reset_intermediates() -> None:
    for path in (BOOKS_FILE, FAIL_FILE, LIST_META_FILE, PROGRESS_FILE, RUN_META_FILE):
        if path.exists():
            path.unlink()
    for path in (NAMES_FILE, CLEAN_FILE, STATS_FILE, REPORT_FILE):
        if path.exists():          # 中间文件清空时交付文件也一并清掉，避免新旧混用
            path.unlink()
    print("[reset] 已清空 data/ 下中间文件与旧交付文件")


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="抓取晋江文学城三类频道主角名（百合雷达数据集）")
    p.add_argument("--pages", type=int, default=PAGES_PER_VIEW, help="每个列表视图抓多少页（默认 10）")
    p.add_argument("--views", type=int, default=len(VIEWS), help="主视图数（默认 2：已完结/连载中）")
    p.add_argument("--extra", type=int, default=len(EXPANSION_VIEWS),
                   help="最多追加几个扩量视图（默认全部 %d 个）" % len(EXPANSION_VIEWS))
    p.add_argument("--target", type=int, default=TARGET_UNIQUE_PER_LABEL,
                   help="每类清洗后唯一名目标（默认 %d，留出跨类歧义剔除余量）" % TARGET_UNIQUE_PER_LABEL)
    p.add_argument("--limit", type=int, default=0, help="本次每类最多新抓书数（0=不限，试跑用）")
    p.add_argument("--workers", type=int, default=WORKERS, help="并发数（契约 ≤4）")
    p.add_argument("--max-seconds", type=int, default=MAX_SECONDS, help="总时长软上限（秒）")
    p.add_argument("--rebuild", action="store_true", help="不联网，仅从中间文件重建交付文件")
    p.add_argument("--reset", action="store_true", help="清空中间文件后重抓")
    p.add_argument("--no-resume", action="store_true", help="忽略断点，重抓所有书")
    args = p.parse_args(argv)
    if args.workers > 4:
        print("[warn] 并发超过契约上限 4，已收敛为 4")
        args.workers = 4
    return args


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    t0 = time.time()
    if args.reset:
        reset_intermediates()

    if args.rebuild:
        done, books, fail_counts = load_done()
        if not books:
            print("没有可用的中间文件 data/_books.jsonl，请先运行抓取。")
            return 2
        list_meta = json.loads(LIST_META_FILE.read_text(encoding="utf-8")) if LIST_META_FILE.exists() else {}
        run_meta = load_run_meta()
        stats = build_outputs(list_meta, books, fail_counts,
                              {"requests": 0, "detail_requests": 0, "bytes_read": 0,
                               "retried": 0, "forbidden": 0},
                              time.time() - t0, args, run_meta)
        print_stats(stats)
        return 0

    fetcher = Fetcher()
    run_meta = load_run_meta()
    done, books, fail_counts = load_done()
    if args.no_resume:
        done.clear()
        books.clear()
        save_progress(done, books, t0)
        print("[no-resume] 忽略断点，重新抓取所有书")
    elif done:
        print(f"[resume] 已从中间文件恢复 {len(done)} 本已处理书目（成功 {len(books)}），继续抓取")
    label_unique: dict[str, set[str]] = {}
    for row in books.values():
        label_unique.setdefault(row["label"], set()).update(row["names"])

    print("[1/3] 抓取列表页 …")
    list_meta = collect_list_phase(fetcher, args.pages, args.views)
    list_requests = fetcher.requests
    LIST_META_FILE.write_text(json.dumps(list_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[2/3] 抓取详情页并解析主角名 …")
    queues = {}
    for lab in LABEL_ORDER:
        seen = set()
        q = []
        for v in list_meta[lab]["views"]:
            for nid in v["ids"]:
                if nid not in seen:
                    seen.add(nid)
                    q.append(nid)
        queues[lab] = q
        print(f"  {lab}: 候选 {len(q)} 本，已有唯一名 {len(label_unique.get(lab, ()))}")
    process_details(fetcher, queues, done, books, fail_counts, label_unique, args, t0)

    # 主视图不够 → 按 EXPANSION_VIEWS 顺序追加扩量视图，直到每类达到目标 / 视图用尽 / 超时
    expansion_used = []
    for vi, view in enumerate(EXPANSION_VIEWS[: args.extra]):
        short = [lab for lab in LABEL_ORDER if len(label_unique.get(lab, ())) < args.target]
        if not short or fetcher.aborted:
            break
        if time.time() - t0 > args.max_seconds * 0.92:
            print("  [stop] 已达总时长软上限，停止追加扩量视图")
            break
        print(f"[2b] 扩量视图 {view['name']}（{vi + 1}/{min(args.extra, len(EXPANSION_VIEWS))}）"
              f"→ {short}")
        added_total = 0
        for lab in short:
            ids = append_extra_view(fetcher, list_meta, lab, view, args.pages)
            seen = set(queues[lab])
            new_ids = [i for i in ids if i not in seen and i not in done]
            added_total += len(new_ids)
            queues[lab].extend(new_ids)
            print(f"    {lab}: 视图 {len(ids)} 本，其中新书 {len(new_ids)} 本，已排队")
        expansion_used.append({"name": view["name"], "new_books": added_total})
        LIST_META_FILE.write_text(json.dumps(list_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if added_total == 0:
            continue
        process_details(fetcher, queues, done, books, fail_counts, label_unique, args, t0)
        print(f"  当前唯一名："
              + ", ".join(f"{lab}={len(label_unique.get(lab, ()))}" for lab in LABEL_ORDER))

    print("[3/3] 生成交付文件 …")
    fetcher_stats = {"requests": fetcher.requests, "bytes_read": fetcher.bytes_read,
                     "retried": fetcher.retried, "forbidden": fetcher.forbidden,
                     "aborted": fetcher.aborted, "abort_reason": fetcher.abort_reason,
                     "detail_requests": max(0, fetcher.requests - list_requests)}
    elapsed = time.time() - t0
    run_meta = update_run_meta(run_meta, fetcher_stats, elapsed)
    args.expansion_used = expansion_used
    stats = build_outputs(list_meta, books, fail_counts, fetcher_stats, elapsed, args, run_meta)
    save_progress(done, books, t0)
    print_stats(stats)
    if fetcher.aborted:
        print(f"[abort] 抓取被熔断：{fetcher.abort_reason}")
        return 3
    return 0


def print_stats(stats: dict) -> None:
    print(f"  by_label: {stats['by_label']}   ambiguous_names: {stats['ambiguous_names']}")
    print(f"  books: {stats['books']}   pages_per_label: {stats['pages_per_label']}")
    print(f"  raw_occurrences: {stats['raw_name_occurrences']}   "
          f"raw_unique: {stats['raw_unique_names']}")
    print(f"  names.jsonl={stats['names_jsonl_rows']} 行，"
          f"names_clean.jsonl={stats['names_clean_jsonl_rows']} 行")


if __name__ == "__main__":
    raise SystemExit(main())

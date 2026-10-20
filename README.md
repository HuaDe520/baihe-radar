# 百合雷达 Baihe Radar 🏳️‍🌈

> 输入任意名字，用一个小神经网络做「名字雷达」：性别倾向、ACG 属性、作品频道倾向、百合指数。
>
> **纯娱乐向玩具。** 训练数据是真实姓名语料 + 虚构角色/小说数据；不能也不应用于判断任何真实个人的性取向或性别身份。

**在线体验**：<https://huade520.github.io/baihe-radar/> ｜ **仓库**：<https://github.com/HuaDe520/baihe-radar>

![百合雷达预览](docs/preview.png)

## 它输出什么

输入一个名字（中文、英文、音译都行），浏览器本地跑一个字符级多头 CNN：

| 输出 | 头 | 数据 | 说明 |
| --- | --- | --- | --- |
| **性别倾向** 女/男 | gender (2 类) | 真实姓名语料 8 万条（中文 4 万 + 外国名/音译 4 万） | 「这个名字的用字像女名/男名」，来自真实姓名统计，不是性别身份判断 |
| **ACG 属性** 伪娘/扶她/伪郎/普通 | expression (4 类) | 萌娘百科角色分类 ~800 条 | 只对二次元角色名有参考价值，数据少，实验性 |
| **作品倾向** 百合/纯爱/言情 | genre (3 类) | 晋江文学城主角名 18,113 条 | 「这个名字更像哪个频道的主角」，虚构作品统计 |
| **百合指数** | yuri (2 类) | 同上 | 页面大圆环 |

外加**逐字贡献**（遮挡实验，可切换看对哪个头的贡献）、模型卡（每头准确率/平衡准确率/样本数）、免责声明。全部推理在浏览器完成，无后端、不收集输入。

## 数据从哪来

- **真实姓名（性别）**：[Chinese-Names-Corpus](https://github.com/wainshine/Chinese-Names-Corpus)
  - 中文姓名带性别 120 万条，抽样 4 万；外国名（中文音译 + 拉丁原名 + M/F）约 46 万条，抽样 4 万
- **ACG 性别表达（伪娘/扶她/伪郎/普通）**：萌娘百科公开分类页，用 `data/scrape_moegirl.py` 抓取（Playwright 过 Cloudflare）
- **作品频道 / 百合判定**：晋江文学城作品库三个频道（`xx0=3/2/1`）的公开「主角名」字段，11,742 本书 → 18,113 个去重主角名（见 `data/DATA_REPORT.md`）

## 模型（v3 四头）

```
字 embedding 32 + 位置 embedding 8（拼接 40 维/位置，max_len=12，支持中英混输）
  → 三路卷积 k=1/2/3（64/96/96 通道）+ BatchNorm → ReLU
  → 每路 max-pool + mean-pool（512 维）
  → 拼首字/末字 embedding（576 维）→ FC 160 → ReLU
  → 4 个分类头：gender / expression / genre / yuri
参数量约 26 万，权重导出 web/model.json（约 1.8 MB），浏览器端 JS 前向推理
（与 numpy 参考实现 4 头误差 < 1e-6，259 条对拍通过）
```

| 指标 | 数值 |
| --- | --- |
| 性别 · 中文名测试准确率 | **95.6%**（4,000 条） |
| 性别 · 外国名/音译测试准确率 | **89.1%**（4,000 条） |
| ACG 属性测试 | acc 66.7% / 平衡准确率 40.2%（仅 78 条测试，数据少，实验性） |
| 作品频道测试准确率 | 43.5%（多数类基线 33.5%） |
| 百合判定平衡准确率 | 53.8% |

## 为什么不做「真实公众人物的性取向概率」？

这是刻意不做的：性取向/性别认同属于敏感个人信息，真实人物没有可靠的公开标注，模型预测既不准也容易造成「被出柜」与歧视；「名字自我实现」的说法在真实世界主要体现为**性别**（父母按性别取名、本人按性别认同改名），这部分我们用真实姓名语料做了可验证的性别头（95.6%/89.1%）。ACG 属性只对虚构角色名做分类；性取向维度只保留虚构小说频道的「作品倾向」，不指向真人。

## 和同类玩具的区别

参考过 [Akanyi/Gaynum](https://github.com/Akanyi/Gaynum)：它是纯前端**确定性哈希**（名字转拼音/罗马音后取字符码做 `(sum×长度×7+首尾字母积+特殊字符和)%101`），README 自己写「结果纯随机，没有任何依据」，没有数据集也没有准确率。
本项目是真的训练了 4 个头（真实姓名 + 萌娘百科 + 晋江），才有这些诚实但不亮眼的数字。

## 目录结构

```
data/    抓取脚本 + 数据集（names*.jsonl / moe_*.jsonl / stats.json / DATA_REPORT.md / PREPARE.md）
train/   训练与导出（train_v3.py / ref_predict_v3.py / make_vectors_v3.py / metrics_v3.json / 早期实验）
web/     静态站点（index.html / style.css / app.js / inference.js / model.json）
docs/    截图
```

## 本地运行

```bash
# 1. 姓名语料（仓库外）
git clone --depth 1 https://github.com/wainshine/Chinese-Names-Corpus.git D:\names-corpus

# 2. 可选：重抓 ACG 角色分类（已有 data/moe_*.jsonl）
python data/scrape_moegirl.py

# 3. 训练 v3 + 导出
python train/train_v3.py --corpus-root D:\names-corpus --out web/model.json

# 4. 对拍 + 预览
python train/make_vectors_v3.py web/model.json
node web/parity_test.mjs
python -m http.server 8000 -d web
```

## 免责声明

本工具纯属娱乐。性别/属性输出只是名字用字的统计倾向，不能用于判断任何真实个人的性取向或性别身份；
真人请尊重其本人自我认同。ACG 属性只对二次元角色名有参考价值。

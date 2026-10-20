# 百合雷达 Baihe Radar 🏳️‍🌈

> 输入任意名字，用一个小神经网络测测它的「百合指数」。
>
> **纯娱乐向玩具。** 训练数据全部是虚构小说里的主角名，模型学到的只是网络小说频道的取名风格，
> 不能也不应用于判断任何真实个人的性取向。

**在线体验**：<https://huade520.github.io/baihe-radar/> ｜ **仓库**：<https://github.com/HuaDe520/baihe-radar>

![百合雷达预览](docs/preview.png)

## 它做什么

输入一个名字（比如「林晚」），浏览器本地跑一个字符级双头 CNN，输出：

- **百合指数**：二分类头给出的 P(百合)，页面大圆环就是它
- **三类概率**（辅助参考）：三分类头给出的 百合 / 纯爱(BL) / 言情(BG)
- **逐字贡献**：把每个字替换成「未知字」重算，看它给百合指数加了多少分（遮挡实验）
- 模型卡：训练样本数、三分类测试准确率、百合判定平衡准确率、随机基线

全部推理在浏览器里完成，没有后端、没有接口请求、不上传你输入的任何内容。

## 数据是怎么来的

- 来源：**晋江文学城作品库**（`jjwxc.net`）公开页面，按频道取「百合 / 纯爱 / 言情」三个分类：
  - 列表页：`https://www.jjwxc.net/bookbase.php?...&xx0={3|2|1}...&page=N`（`xx0=3` 百合、`2` 纯爱、`1` 言情）
  - 详情页：`https://www.jjwxc.net/onebook.php?novelid=ID`，只读公开的 `<meta name="Keywords">` 里的
    `主角：...` 字段，并用页面里的 `itemprop="genre"`（文章类型）校验频道关键词，防止串类
- 规模：每个频道 30 个列表页（主视图 + 积分排序 / 字数排序 / 免费库扩量视图），共解析 **11,742 本书**，
  得到 **百合 6,047 / 纯爱(BL) 6,075 / 言情(BG) 5,991** 个去重主角名，合计 **18,113** 条训练样本；
  跨频道同名（541 个）全部剔除
- 只收 2–6 个汉字（允许间隔号 `·`）的名字，过滤「众人」「路人甲」等非人名；全程并发 ≤4 + 随机延时，未触发封禁
- 抓取脚本与原始数据都在仓库里：`data/scrape.py`、`data/names.jsonl`、`data/DATA_REPORT.md`

## 模型（v2 双头）

字符级 CNN，从零训练（PyTorch），共享编码器 + 两个输出头：

```
字 embedding 32 维 + 位置 embedding 8 维（拼接 = 40 维/位置）
  → 三路卷积 k=1/2/3（64/96/96 通道） + BatchNorm → ReLU
  → 每路 max-pool + mean-pool（共 512 维）
  → 拼首字 / 末字 embedding（各 32 维 → 576 维）
  → FC 128 → ReLU
  → 头 A：3 分类 softmax（百合 / 纯爱 / 言情）
  → 头 B：2 分类 softmax（百合 vs 非百合）→ 页面「百合指数」
参数量约 16 万，权重导出成 web/model.json（约 1.7 MB），浏览器端 JS 直接前向推理
（与 Python/numpy 参考实现误差 < 1e-6）
```

训练：14,489 / 1,812 / 1,812（train/val/test 按名字分层切分），Adam + label smoothing + 早停 + dropout 0.4。

| 指标 | 数值 |
| --- | --- |
| 三分类测试准确率 | **45.0%**（多数类基线 33.5%） |
| 三分类 macro-F1 | 0.438 |
| 二分类（百合 vs 非百合）准确率 | 66.8% |
| **二分类平衡准确率（主力指标）** | **58.1%** |
| 二分类 macro-F1 | 0.581 |

## 为什么准确率只有 45% / 58%？

因为「名字」本身携带的信息就这么点，不是模型没调好。我们试过：

- 更大更宽的 CNN、加深一层、BatchNorm、dropout 0.3–0.5、label smoothing、位置嵌入、首末字特征、BiGRU、
  多任务双头（三分类 + 二分类辅助）——三分类准确率在 42%~46% 之间晃动，**架构不是瓶颈**；
- 数据从 7,370 条扩到 18,113 条（2.5 倍），三分类从 42.2% 提到 45.0%，**数据量有帮助，但只值几个点**；
- 3 seed 集成只能再多 ~1 个点，不值得让浏览器多下 3 MB 权重。

根源是标签本身的统计结构：百合频道主角基本是女名，纯爱频道基本是男名，言情频道男女都有，
所以模型很大程度只是在学「名字像不像女生」；而「百合 vs 言情女主」这种真正想区分的风格差异，在名字上非常微弱。
**当成占卜玩具就好，别当成鉴人工具。**

## 为什么跑在浏览器里？

GitHub Pages 只能托管静态文件，跑不了 Python 后端；而这个模型只有 1.7 MB，实测（Node/Chromium）：

- 加载并构建权重 ~6–10 ms
- 单次预测 ~2 ms
- 一次逐字遮挡归因 ~2 ms

放浏览器里完全不为难，反而免了服务器和隐私问题。真要上大模型（比如几 MB~几十 MB 的 Transformer），
也可以继续用 ONNX Runtime Web / WebGPU 在浏览器跑，或者另起后端；当前这个体量没必要。

## 和同类玩具的区别

参考过 [Akanyi/Gaynum](https://github.com/Akanyi/Gaynum)（名字 Gay 指数）：它是纯前端**确定性哈希**——
名字转拼音/罗马音，取字符码做一套 `(sum × 长度 × 7 + 首尾字母积 + 特殊字符和) % 101` 的运算，
再随机抽一条话术。README 里也写着「结果纯随机，没有任何依据」。它没有训练、没有数据集、没有准确率。
本项目是**真的用 1.8 万个带频道标签的名字训练出来的分类器**，所以才有「45% 准确率」这种诚实但不亮眼的数字。

## 目录结构

```
data/    抓取脚本 + 数据集（names.jsonl / names_clean.jsonl / stats.json / DATA_REPORT.md）
train/   训练与导出脚本（train_v2.py / ref_predict_v2.py / make_vectors_v2.py / metrics_v2.json / exp*.py）
web/     静态站点（index.html / style.css / app.js / inference.js / model.json）
docs/    截图
```

## 本地运行

```bash
# 1. 抓数据（断点续抓，--rebuild 可离线重建）
python data/scrape.py

# 2. 训练 + 导出 web/model.json
python train/train_v2.py --data data/names_clean.jsonl --out web/model.json

# 3. 生成对拍向量，验证 JS 与 Python 推理一致
python train/make_vectors_v2.py web/model.json
node web/parity_test.mjs

# 4. 本地预览
python -m http.server 8000 -d web
```

## 免责声明

本工具纯属娱乐，训练数据仅为虚构小说角色名，不能也不应用于判断任何真实个人的性取向。
如果你在真实生活里好奇某个人的取向，最好的办法是尊重对方、等 ta 自己说。

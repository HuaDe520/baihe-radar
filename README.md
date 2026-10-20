# 百合雷达 Baihe Radar

> 输入任意名字，用一个小神经网络测测它的「百合浓度」。
> **纯娱乐向玩具**：训练数据全部是虚构小说主角的名字，模型学到的只是网络小说频道的取名风格，
> 不能也不应用于判断任何真实个人的性取向。

## 这是什么

- 数据：从**晋江文学城作品库**（公开页面）抓取「百合 / 纯爱(BL) / 言情(BG)」三个频道的小说主角名，
  仅抓取公开的 `<meta name="Keywords">` 中的「主角：」字段。
- 模型：字符级卷积神经网络（embedding 28 维 + k=1/2/3 三路卷积 + ReLU + global max-pool + FC + 3 分类），
  从零训练，纯 Python/PyTorch 脚本，导出成 `web/model.json` 后由浏览器端 JS 直接推理（无后端）。
- 网站：纯静态页面（GitHub Pages 托管），粉紫玻璃拟态风，输入名字后输出
  百合 / 纯爱(BL) / 言情(BG) 三类概率、百合浓度，以及逐字的遮挡贡献。

## 目录结构

```
data/   抓取脚本与数据集（names.jsonl / names_clean.jsonl / stats.json / DATA_REPORT.md）
train/  训练与导出脚本（train.py / ref_predict.py / make_vectors.py）
web/    静态站点（index.html / style.css / app.js / inference.js / model.json）
```

## 本地运行

```bash
# 抓数据（断点续抓）
python data/scrape.py
# 训练 + 导出 web/model.json
python train/train.py
# 生成对拍向量
python train/make_vectors.py
# JS 与 Python 推理对拍
node web/parity_test.mjs
# 本地预览
python -m http.server 8000 -d web
```

## 免责声明

本工具纯属娱乐，训练数据仅为虚构小说角色名，不能也不应用于判断任何真实个人的性取向。

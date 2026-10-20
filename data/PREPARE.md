# 数据准备（v6）

v6 模型用了五份数据，其中前两份是「真实世界」的姓名统计，后三份是虚构作品/角色：

| 头 | 数据 | 规模 | 来源 |
| --- | --- | --- | --- |
| gender | 中文姓名（带性别） | 40,000（抽样） | [Chinese-Names-Corpus](https://github.com/wainshine/Chinese-Names-Corpus) `Chinese_Names_Corpus_Gender（120W）.txt` |
| gender | 外国姓名（音译 + 拉丁原名，带性别） | 40,000（抽样） | 同上 `English_Names_Corpus/English_Cn_Name_Corpus_Gender（48W）.txt` |
| gender | 日文名（日文原写法 + 罗马音，带性别） | 4,978 | [VNDB](https://vndb.org) 公开角色 API（`data/scrape_vndb.py` + `data/build_vndb_data.py`），单独成组 `gender_jp` |
| expression | 伪娘 / 扶她 / 伪郎 / 普通角色 | 1,054 | 萌娘百科分类页（`data/scrape_moegirl.py`） |
| yuri_char | 百合角色 202（萌娘 177 + VNDB 25） vs 其他角色 519 | 721 | 萌娘百科 `分类:百合` + `按生日分类` 随机角色 + VNDB 百合向 VN 角色 |
| genre / yuri | 晋江作品库主角名 | 18,113 | `data/scrape.py`，见 `DATA_REPORT.md` |

网名支持：训练时对部分样本追加 `_233 / 233 / _01 / 666 / 77 / x` 后缀做增广，并在词表里加入数字/下划线。
日文支持：VNDB 日文原写法（假名/汉字）与罗马音都进了词表与 `gender_jp` 训练组。

## 取向转移矩阵（数据估计）

`train/learn_orientation_priors.py` 从两份公开数据统计 4×4 转移矩阵：

1. 晋江作品库中「主角」名字 ≥2 个的书，构成配对边；
   频道给类型：百合→女同、纯爱→男同、言情→按性别头预测男女；
   清洗：言情只保留预测一男一女的配对，百合/纯爱只保留预测同性的配对；
2. VNDB 中同一 VN 的角色构成配对边，VN 标签给类型（Yuri/Lesbian、BL、Romance）。

输出 `web/orientation_priors.json`：counts、纯数据 matrix_raw、以及 `counts + 300 × 内置弱先验` 平滑后的 matrix（页面使用）。当前规模：jjwxc 11,274 对 + VNDB 2,134 对 = **13,408 对**。

> 说明：公开数据里几乎不存在「男同喜欢直男」这类跨类型关系的可靠标签，所以矩阵主结构是同向/直向，跨类型小边来自弱先验平滑——页面会如实显示来源与样本数。

## 复现步骤

```bash
# 1. 仓库外准备姓名语料（约 33MB）
git clone --depth 1 https://github.com/wainshine/Chinese-Names-Corpus.git D:\names-corpus

# 2. ACG 角色分类（萌娘百科，需要 playwright + 本机 Edge；仓库已带 data/moe_*.jsonl 可跳过）
python data/scrape_moegirl.py

# 3. VNDB 日文角色（仓库已带 data/vndb_*.jsonl 可跳过）
python data/scrape_vndb.py
python data/build_vndb_data.py

# 4. 训练 v6 五头模型
python train/train_v6.py --corpus-root D:\names-corpus --out web/model.json

# 5. 数据学转移矩阵
python train/learn_orientation_priors.py

# 6. 对拍
python train/make_vectors_v6.py web/model.json
node web/parity_test.mjs
```

> 性别头是真实姓名的统计用字规律，不是身份判断；性取向不针对真人；ACG 角色标签来自公开分类页。

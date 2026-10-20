# 数据准备（v4）

v4 模型用了五份数据，其中前两份是「真实世界」的姓名统计，后三份是虚构作品/角色：

| 头 | 数据 | 规模 | 来源 |
| --- | --- | --- | --- |
| gender | 中文姓名（带性别） | 40,000（抽样） | [Chinese-Names-Corpus](https://github.com/wainshine/Chinese-Names-Corpus) `Chinese_Names_Corpus_Gender（120W）.txt` |
| gender | 外国姓名（音译 + 拉丁原名，带性别） | 40,000（抽样） | 同上 `English_Names_Corpus/English_Cn_Name_Corpus_Gender（48W）.txt` |
| expression | 伪娘 / 扶她 / 伪郎 / 普通角色 | 1,054 | 萌娘百科分类页（`data/scrape_moegirl.py`） |
| yuri_char | 百合角色 177 vs 其他角色 519 | 696 | 萌娘百科 `分类:百合` + `按生日分类` 随机角色 |
| genre / yuri | 晋江作品库主角名 | 18,113 | `data/scrape.py`，见 `DATA_REPORT.md` |

网名支持：训练时对部分样本追加 `_233 / 233 / _01 / 666 / 77 / x` 后缀做增广，并在词表里加入数字/下划线。

## 复现步骤

```bash
# 1. 仓库外准备姓名语料（约 33MB）
git clone --depth 1 https://github.com/wainshine/Chinese-Names-Corpus.git D:\names-corpus

# 2. 抓 ACG 角色分类（需要 playwright + 本机 Edge；仓库已带 data/moe_*.jsonl 可跳过）
python data/scrape_moegirl.py

# 3. 训练 v4 五头模型
python train/train_v4.py --corpus-root D:\names-corpus --out web/model.json

# 4. 对拍
python train/make_vectors_v4.py web/model.json
node web/parity_test.mjs
```

> 性别头是真实姓名的统计用字规律，不是身份判断；性取向不针对真人；ACG 角色标签来自公开分类页。

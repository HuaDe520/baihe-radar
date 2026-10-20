# 数据准备（v3）

v3 模型用了四份数据，其中前两份是「真实世界」的姓名统计，后两份是虚构作品/角色：

| 头 | 数据 | 规模 | 来源 |
| --- | --- | --- | --- |
| 性别 gender | 中文姓名语料（带性别） | 40,000（抽样） | [Chinese-Names-Corpus](https://github.com/wainshine/Chinese-Names-Corpus) 的 `Chinese_Names_Corpus_Gender（120W）.txt`（`name,性别`） |
| 性别 gender | 外国姓名（中文音译 + 拉丁原名，带性别） | 40,000（抽样） | 同上仓库 `English_Names_Corpus/English_Cn_Name_Corpus_Gender（48W）.txt`（`音译|原名|M/F`） |
| ACG 属性 expression | 萌娘百科角色分类（伪娘 / 扶她 / 伪郎 / 普通角色） | ~800 | [萌娘百科](https://zh.moegirl.org.cn) 公开分类页，用 `data/scrape_moegirl.py` 抓取 |
| 作品频道 genre / 百合判定 yuri | 晋江文学城作品库主角名 | 18,113 | 见 `data/scrape.py` + `DATA_REPORT.md` |

## 复现步骤

```bash
# 1. 仓库外准备姓名语料（约 33MB）
git clone --depth 1 https://github.com/wainshine/Chinese-Names-Corpus.git D:\names-corpus

# 2. 抓 ACG 角色分类（需要 playwright + 本机 Edge；已有 data/moe_*.jsonl 可跳过）
python data/scrape_moegirl.py

# 3. 训练 v3 四头模型
python train/train_v3.py --corpus-root D:\names-corpus --out web/model.json

# 4. 对拍
python train/make_vectors_v3.py web/model.json
node web/parity_test.mjs
```

> 说明：性别头是**真实姓名统计**出来的「名字用字像男/像女」的概率，不是性别身份判断；
> 性取向不针对真实个人做预测。ACG 属性只对二次元角色名有参考价值。全部纯娱乐。

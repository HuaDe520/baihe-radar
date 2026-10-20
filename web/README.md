# 百合雷达 BAIHE RADAR · 前端（web/）

纯静态站点：无框架、无构建步骤、无外部 CDN / 字体 / 图片依赖。
模型推理全部在浏览器本地完成，输入不会上传。

## 文件

| 文件 | 说明 |
| --- | --- |
| `index.html` | 页面结构（标题 / 输入 / 结果 / 模型卡 / 免责声明） |
| `style.css` | 粉紫渐变 + 玻璃拟态、移动端优先 |
| `app.js` | 交互与渲染（ES module，导出可单测的纯函数） |
| `inference.js` | 字符级 CNN 推理引擎，严格复刻 `CONTRACT.md` 第二节 |
| `model.json` | **正式权重，由 Lead 训练导出覆盖**；本地调试可用 mock 临时生成 |
| `mock-model.json` | 随机权重（V=120，形状符合契约），仅用于前端调试 |
| `make_mock_model.mjs` | 生成 `mock-model.json` |
| `parity_test.mjs` | JS ↔ Python 参考实现对拍（读 `web/model.json` + `train/test_vectors.json`） |
| `package.json` | 只为让 Node 把 `web/*.js` 当 ES module（浏览器不需要它） |

## 本地预览

```bash
# 1. 没有正式 model.json 时，先用随机权重调试
node web/make_mock_model.mjs
cp web/mock-model.json web/model.json        # Windows: copy web\mock-model.json web\model.json

# 2. 起一个静态服务器（不能用 file:// 直接双击，fetch 会被浏览器拦）
cd web && python -m http.server 8099
# 或：npx serve web

# 3. 打开 http://127.0.0.1:8099/
```

正式权重就位后，直接覆盖 `web/model.json` 即可，前端无需改代码。

## 对拍测试

```bash
node web/parity_test.mjs          # 读 web/model.json + train/test_vectors.json
node web/parity_test.mjs --mock   # 用 mock-model.json 走一遍链路（无真模型时）
```

- `train/test_vectors.json` 尚未生成时，脚本打印「待 lead 生成」并以退出码 0 跳过。
- 逐条比对 `max |Δprobs|`，阈值 `5e-4`（契约第二节）。
- 也可用环境变量 `BAIHE_VECTORS=/path/to/vectors.json` 指定其它向量文件。

## JS 接口

```js
import { loadModel, predict, occlusionAttribution } from './inference.js';

const model = await loadModel('./model.json'); // 浏览器 fetch + 构建权重
const result = predict(model, '林晚');
// result.probs → [pBaihe, pPurelove, pRomance]，和为 1
// result.chars → 实际参与推理的字符（≤8）；result.ids → 补 pad 后的 8 个 id
// result.truncated / result.unknownChars → 超长、未登录字提示

const attribution = occlusionAttribution(model, '林晚');
// [{ index, char, delta, probWithout }]，delta = P(百合|原输入) − P(百合|该字换成 <unk>)
// delta > 0 表示这个字在给百合浓度加分
```

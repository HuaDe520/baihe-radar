/**
 * 百合雷达 BAIHE RADAR · 前端推理引擎（ES module）
 *
 * 严格复刻 CONTRACT.md 第二节的推理算法：
 *   去空白 → 按 Unicode 码点取前 max_len 个字符 → vocab 映射（未登录字用 unk_id）
 *   → 不足补 pad_id → emb 查表 → 三路 conv(k=1/2/3, 行主序 W[c][j*28+e])
 *   → ReLU 后 max-pool → 拼接 112 维 → fc(64) + ReLU → out(3) → softmax
 *
 * 对外接口：
 *   loadModel(url)                      → Promise<Model>  （浏览器里 fetch 模型 JSON）
 *   buildModel(json)                    → Model           （同步构建，Node 对拍/测试用）
 *   predict(model, rawInput)            → {probs, chars, ids, truncated, inputLength, unknownChars}
 *   occlusionAttribution(model, raw)    → [{index, char, delta, probWithout}, ...]
 *   normalizeInput(raw)                 → string           （去空白；空输入抛「请输入名字」）
 *
 * 模型 JSON 的契约见 CONTRACT.md 第二节。本文件只做推理，不包含任何 UI。
 */

/** 契约规定的权重格式标识（仅用于展示，不强制校验，方便 mock 调试）。 */
export const MODEL_FORMAT = 'baihe-charcnn-v1';

function fail(message) {
  throw new Error(message);
}

/** 把普通数组（或 TypedArray）读成 Float64Array，并校验长度与有限性。 */
function readF64(source, expectedLength, label) {
  if (!source || typeof source.length !== 'number') {
    fail(`模型权重 ${label} 缺失`);
  }
  if (source.length !== expectedLength) {
    fail(`模型权重 ${label} 形状不符：期望 ${expectedLength} 个数，实际 ${source.length} 个`);
  }
  const out = new Float64Array(expectedLength);
  for (let i = 0; i < expectedLength; i += 1) {
    const value = Number(source[i]);
    if (!Number.isFinite(value)) {
      fail(`模型权重 ${label} 第 ${i} 个数不是有效数字`);
    }
    out[i] = value;
  }
  return out;
}

function readInt(value, label, { min = 0 } = {}) {
  const n = Number(value);
  if (!Number.isInteger(n) || n < min) {
    fail(`模型 config.${label} 非法：${String(value)}`);
  }
  return n;
}

function shapeOf(source) {
  return source && Array.isArray(source.shape) ? source.shape : null;
}

function checkShape(source, expected, label) {
  const shape = shapeOf(source);
  if (!shape) return; // mock 或简化导出可能省略 shape，按 config 推导即可
  const same = shape.length === expected.length && shape.every((v, i) => Number(v) === expected[i]);
  if (!same) {
    fail(`模型权重 ${label} 的 shape 不符：期望 [${expected.join(', ')}]，实际 [${shape.join(', ')}]`);
  }
}

/**
 * 由模型 JSON 构建可复用的推理模型对象（同步）。
 *
 * @param {object} json  web/model.json 的内容
 * @returns {object}     编译后的模型（Float64Array 权重 + chars 索引表）
 */
export function buildModel(json) {
  if (!json || typeof json !== 'object') {
    fail('模型 JSON 为空或格式错误');
  }
  const config = json.config;
  if (!config || typeof config !== 'object') {
    fail('模型缺少 config');
  }

  const maxLen = readInt(config.max_len, 'max_len', { min: 1 });
  const embDim = readInt(config.emb_dim, 'emb_dim', { min: 1 });
  const hidden = readInt(config.hidden, 'hidden', { min: 1 });
  const numClasses = readInt(config.num_classes, 'num_classes', { min: 1 });
  const padId = readInt(config.pad_id, 'pad_id', { min: 0 });
  const unkId = readInt(config.unk_id, 'unk_id', { min: 0 });

  const classes = Array.isArray(config.classes) ? config.classes.slice() : [];
  if (classes.length !== numClasses) {
    fail(`模型 config.classes 数量不符：期望 ${numClasses} 个，实际 ${classes.length} 个`);
  }
  const classNamesZh =
    Array.isArray(config.class_names_zh) && config.class_names_zh.length === numClasses
      ? config.class_names_zh.slice()
      : classes.slice();

  if (!Array.isArray(json.vocab) || json.vocab.length < 2) {
    fail('模型 vocab 缺失或过短');
  }
  const vocab = json.vocab.slice();
  const vocabSize = vocab.length;
  if (padId >= vocabSize || unkId >= vocabSize) {
    fail('模型 config.pad_id / unk_id 超出 vocab 范围');
  }

  // 与 Array.prototype.indexOf 语义一致：同一个字符取首次出现的位置。
  const charIndex = new Map();
  for (let i = 0; i < vocab.length; i += 1) {
    const token = vocab[i];
    if (!charIndex.has(token)) charIndex.set(token, i);
  }

  const convSpec = Array.isArray(config.conv) ? config.conv : null;
  if (!convSpec || convSpec.length === 0) {
    fail('模型 config.conv 缺失');
  }

  const weights = json.weights;
  if (!weights || typeof weights !== 'object') {
    fail('模型缺少 weights');
  }
  if (!weights.emb || !weights.emb.data) {
    fail('模型权重 emb 缺失');
  }
  checkShape(weights.emb, [vocabSize, embDim], 'emb');
  const emb = readF64(weights.emb.data, vocabSize * embDim, 'emb');

  if (!Array.isArray(weights.conv) || weights.conv.length !== convSpec.length) {
    fail(`模型权重 conv 数量不符：期望 ${convSpec.length} 个分支`);
  }

  const conv = [];
  let hiddenIn = 0;
  for (let i = 0; i < convSpec.length; i += 1) {
    const spec = convSpec[i] || {};
    const k = readInt(spec.k, `conv[${i}].k`, { min: 1 });
    const ch = readInt(spec.ch, `conv[${i}].ch`, { min: 1 });
    if (k > maxLen) {
      fail(`模型 config.conv[${i}].k=${k} 大于 max_len=${maxLen}`);
    }
    const branch = weights.conv[i];
    if (!branch || !branch.W || !branch.W.data || !branch.b || !branch.b.data) {
      fail(`模型权重 conv[${i}] 的 W/b 缺失`);
    }
    checkShape(branch.W, [ch, k * embDim], `conv[${i}].W`);
    checkShape(branch.b, [ch], `conv[${i}].b`);
    conv.push({
      k,
      ch,
      W: readF64(branch.W.data, ch * k * embDim, `conv[${i}].W`),
      b: readF64(branch.b.data, ch, `conv[${i}].b`),
    });
    hiddenIn += 2 * ch;
  }

  if (!weights.fc || !weights.fc.W || !weights.fc.W.data || !weights.fc.b || !weights.fc.b.data) {
    fail('模型权重 fc 缺失');
  }
  checkShape(weights.fc.W, [hidden, hiddenIn], 'fc.W');
  checkShape(weights.fc.b, [hidden], 'fc.b');
  const fcW = readF64(weights.fc.W.data, hidden * hiddenIn, 'fc.W');
  const fcB = readF64(weights.fc.b.data, hidden, 'fc.b');

  if (!weights.out || !weights.out.W || !weights.out.W.data || !weights.out.b || !weights.out.b.data) {
    fail('模型权重 out 缺失');
  }
  checkShape(weights.out.W, [numClasses, hidden], 'out.W');
  checkShape(weights.out.b, [numClasses], 'out.b');
  const outW = readF64(weights.out.W.data, numClasses * hidden, 'out.W');
  const outB = readF64(weights.out.b.data, numClasses, 'out.b');

  let baiheIndex = classes.indexOf('baihe');
  if (baiheIndex < 0) baiheIndex = 0;

  return {
    format: typeof json.format === 'string' ? json.format : MODEL_FORMAT,
    config: {
      max_len: maxLen,
      emb_dim: embDim,
      hidden,
      num_classes: numClasses,
      pad_id: padId,
      unk_id: unkId,
      conv: convSpec.map((spec, i) => ({ k: conv[i].k, ch: conv[i].ch })),
      classes,
      class_names_zh: classNamesZh,
    },
    vocab,
    vocabSize,
    charIndex,
    emb,
    conv,
    hiddenIn,
    fcW,
    fcB,
    outW,
    outB,
    numClasses,
    hidden,
    baiheIndex,
    metrics: json.metrics && typeof json.metrics === 'object' ? json.metrics : null,
    dataset: json.dataset && typeof json.dataset === 'object' ? json.dataset : null,
  };
}

/**
 * 读取模型 JSON 并构建模型。
 * @param {string} url 模型地址，默认同目录 ./model.json
 * @returns {Promise<object>}
 */
export async function loadModel(url = './model.json') {
  if (typeof fetch !== 'function') {
    fail('当前环境不支持 fetch，无法加载模型');
  }
  let response;
  try {
    response = await fetch(url, { cache: 'no-cache' });
  } catch (err) {
    fail(`无法请求模型文件（${url}）：${err && err.message ? err.message : String(err)}`);
  }
  if (!response.ok) {
    fail(`模型文件加载失败：HTTP ${response.status}（${url}）`);
  }
  let json;
  try {
    json = await response.json();
  } catch (err) {
    fail(`模型文件不是合法 JSON：${url}`);
  }
  return buildModel(json);
}

/**
 * 按契约第 1 步清洗输入：取字符串、去掉所有空白字符（\s）。
 * 清洗后为空 → 抛「请输入名字」。
 */
export function normalizeInput(rawInput) {
  const text = rawInput == null ? '' : String(rawInput);
  const cleaned = text.replace(/\s+/g, '');
  if (!cleaned) fail('请输入名字');
  return cleaned;
}

/**
 * 由字符 id 序列前向计算（内部函数）。
 * 顺序：emb 查表 → 三路卷积（ReLU + max-pool）→ 拼接 → fc + ReLU → out → softmax。
 */
function forwardFromIds(model, ids) {
  const { max_len: L, emb_dim: E } = model.config;
  const hiddenIn = model.hiddenIn;

  // 1) X = ids.map(id => emb[id])，8 行 × E 列
  const X = new Float64Array(L * E);
  for (let i = 0; i < L; i += 1) {
    const id = ids[i];
    if (!Number.isInteger(id) || id < 0 || id >= model.vocabSize) {
      fail(`字符 id 越界：${id}`);
    }
    const src = id * E;
    const dst = i * E;
    for (let e = 0; e < E; e += 1) X[dst + e] = model.emb[src + e];
  }

  // 2) 三路卷积：W[c][j*E + e] 对应窗口第 j 个位置第 e 维；每路同时做 max-pool 与 mean-pool
  const pooled = new Float64Array(hiddenIn);
  let offset = 0;
  for (const branch of model.conv) {
    const { k, ch, W, b } = branch;
    const windows = L - k + 1;
    for (let c = 0; c < ch; c += 1) {
      let best = 0;
      let total = 0;
      const wBase = c * k * E;
      for (let p = 0; p < windows; p += 1) {
        let z = b[c];
        for (let j = 0; j < k; j += 1) {
          const xBase = (p + j) * E;
          const wOff = wBase + j * E;
          for (let e = 0; e < E; e += 1) z += W[wOff + e] * X[xBase + e];
        }
        if (z > best) best = z;
        if (z > 0) total += z;
      }
      pooled[offset + c] = best;                         // max_p ReLU(z)
      pooled[offset + ch + c] = total / windows;         // mean_p ReLU(z)
    }
    offset += 2 * ch;
  }

  // 3) fc: h1 = relu(fc.W · h0 + fc.b)，h0 即 pooled（每路 max+mean，(32+40+40)*2=224 维）
  const h1 = new Float64Array(model.hidden);
  for (let o = 0; o < model.hidden; o += 1) {
    let z = model.fcB[o];
    const wBase = o * hiddenIn;
    for (let i = 0; i < hiddenIn; i += 1) z += model.fcW[wBase + i] * pooled[i];
    h1[o] = z > 0 ? z : 0;
  }

  // 4) out: logits = out.W · h1 + out.b
  const logits = new Float64Array(model.numClasses);
  for (let o = 0; o < model.numClasses; o += 1) {
    let z = model.outB[o];
    const wBase = o * model.hidden;
    for (let i = 0; i < model.hidden; i += 1) z += model.outW[wBase + i] * h1[i];
    logits[o] = z;
  }

  // 5) softmax（减最大值，数值稳定）
  let maxLogit = -Infinity;
  for (let o = 0; o < model.numClasses; o += 1) {
    if (logits[o] > maxLogit) maxLogit = logits[o];
  }
  const probs = new Array(model.numClasses);
  let sum = 0;
  for (let o = 0; o < model.numClasses; o += 1) {
    const value = Math.exp(logits[o] - maxLogit);
    probs[o] = value;
    sum += value;
  }
  for (let o = 0; o < model.numClasses; o += 1) probs[o] /= sum;

  return { probs, logits: Array.from(logits), pooled: Array.from(pooled) };
}

/**
 * 预测一个名字的三分类概率。
 *
 * @param {object} model     buildModel / loadModel 得到的模型
 * @param {string} rawInput  原始输入（可含空白，可超长）
 * @returns {{probs: number[], chars: string[], ids: number[], truncated: boolean,
 *            inputLength: number, unknownChars: string[]}}
 *          probs 顺序与 config.classes 一致（契约固定为 [baihe, purelove, romance]）；
 *          chars 是实际参与推理的字符（≤ max_len）；ids 是补齐到 max_len 的字符 id。
 */
export function predict(model, rawInput) {
  if (!model || !model.emb) fail('模型未加载');
  const cleaned = normalizeInput(rawInput);
  const codePoints = Array.from(cleaned);
  const maxLen = model.config.max_len;
  const truncated = codePoints.length > maxLen;
  const chars = codePoints.slice(0, maxLen);

  const ids = new Array(maxLen).fill(model.config.pad_id);
  const unknownChars = [];
  for (let i = 0; i < chars.length; i += 1) {
    const char = chars[i];
    const id = model.charIndex.get(char);
    if (id === undefined) {
      ids[i] = model.config.unk_id;
      unknownChars.push(char);
    } else {
      ids[i] = id;
    }
  }

  const { probs } = forwardFromIds(model, ids);
  return {
    probs,
    chars,
    ids,
    truncated,
    inputLength: codePoints.length,
    unknownChars,
  };
}

/**
 * 逐字遮挡归因：把第 i 个字符替换成 unk_id 后重新前向，返回百合概率的变化。
 *
 * delta = P(百合 | 原输入) − P(百合 | 该字换成 <unk>)
 *   delta > 0 → 这个字在给百合浓度加分
 *   delta < 0 → 这个字在给百合浓度减分
 *
 * @param {object} model
 * @param {string} rawInput
 * @returns {Array<{index:number, char:string, delta:number, probWithout:number}>}
 *          按输入位置排序，长度 = 实际参与推理的字符数（不含 pad）。
 */
export function occlusionAttribution(model, rawInput) {
  const base = predict(model, rawInput);
  const baiheIndex = model.baiheIndex;
  const baseProb = base.probs[baiheIndex];
  const unkId = model.config.unk_id;

  return base.chars.map((char, index) => {
    const occludedIds = base.ids.slice();
    occludedIds[index] = unkId;
    const occludedProbs = forwardFromIds(model, occludedIds).probs;
    const probWithout = occludedProbs[baiheIndex];
    return {
      index,
      char,
      delta: baseProb - probWithout,
      probWithout,
    };
  });
}

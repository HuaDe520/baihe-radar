/**
 * 百合雷达 BAIHE RADAR · 前端推理引擎 v4（ES module）
 *
 * 严格复刻 `train/train_v4.py`（MultiHead）导出的 baihe-multi-v4 权重：
 *   输入：trim → 去掉所有空白 → ASCII 大写字母转小写 → 按 Unicode 码点取前 max_len(=12) 个字符
 *        （中文名、英文名、网名的字母/数字/下划线等字符都按原样进入 vocab 查表）
 *        → vocab 查表（未登录用 unk_id=1）→ 不足补 pad_id=0 到 12
 *   编码：X[i] = concat(emb[ids[i]] 32 维, pos[i] 8 维) = 每位置 40 维（concat，不是相加）
 *   卷积：三路 conv k=1/2/3（W[c][j*40+e]）→ BatchNorm(gamma,beta,mean,var,bn_eps) → ReLU
 *        → 每路 max + mean 双池化 → 按分支顺序拼 [max, mean] = 512 维
 *   特征：512 + 首字 emb 32 + 末个非 pad 字 emb 32 = 576 → fc(160) + ReLU
 *   五头：gender(2) / expression(4) / genre(3) / yuri(2) / yuri_char(2) 各自 Linear + softmax
 *
 * 纯娱乐向玩具；本文件只做推理，不含任何 UI。
 *
 * 对外接口：
 *   loadModel(url)                                  → Promise<Model>
 *   buildModel(json)                                → Model
 *   predict(model, rawInput)                        → {probs:{gender,expression,genre,yuri,yuri_char},
 *                                                       baiheIndex, chars, ids, truncated,
 *                                                       inputLength, unknownChars}
 *   occlusionAttribution(model, raw, head, classIndex)
 *                                                   → [{index, char, delta, probWithout,
 *                                                       head, classIndex, className}]
 *   normalizeInput(raw)                             → string
 */

/** v4 权重格式标识（强制校验，避免误把旧权重当 v4 加载） */
export const MODEL_FORMAT = 'baihe-multi-v4';

/** 五个头的固定顺序（用于展示与校验） */
export const HEAD_KEYS = ['gender', 'expression', 'genre', 'yuri', 'yuri_char'];

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

function readNumber(value, label, { min = -Infinity } = {}) {
  const n = Number(value);
  if (!Number.isFinite(n) || n < min) {
    fail(`模型 config.${label} 非法：${String(value)}`);
  }
  return n;
}

function checkShape(source, expected, label) {
  const shape = source && Array.isArray(source.shape) ? source.shape : null;
  if (!shape) return; // 简化导出可能省略 shape，按 config 推导即可
  const same = shape.length === expected.length && shape.every((v, i) => Number(v) === expected[i]);
  if (!same) {
    fail(`模型权重 ${label} 的 shape 不符：期望 [${expected.join(', ')}]，实际 [${shape.join(', ')}]`);
  }
}

function readWeightBlock(source, expectedShape, label) {
  if (!source || !source.data) fail(`模型权重 ${label} 缺失`);
  checkShape(source, expectedShape, label);
  const total = expectedShape.reduce((a, b) => a * b, 1);
  return readF64(source.data, total, label);
}

function readBnBlock(block, ch, label) {
  if (!block || typeof block !== 'object') fail(`模型权重 ${label} 缺失`);
  return {
    gamma: readWeightBlock(block.gamma, [ch], `${label}.gamma`),
    beta: readWeightBlock(block.beta, [ch], `${label}.beta`),
    mean: readWeightBlock(block.mean, [ch], `${label}.mean`),
    var: readWeightBlock(block.var, [ch], `${label}.var`),
  };
}

function readHeadConfig(headKey, raw) {
  if (!raw || typeof raw !== 'object') fail(`模型 config.heads.${headKey} 缺失`);
  const classes = Array.isArray(raw.classes) ? raw.classes.slice() : [];
  if (classes.length === 0) fail(`模型 config.heads.${headKey}.classes 为空`);
  const namesZh =
    Array.isArray(raw.names_zh) && raw.names_zh.length === classes.length ? raw.names_zh.slice() : classes.slice();
  return { key: headKey, classes, namesZh, numClasses: classes.length };
}

/**
 * 由模型 JSON 构建可复用的推理模型对象（同步）。
 * @param {object} json  web/model.json 的内容（必须 format = baihe-multi-v3）
 * @returns {object}
 */
export function buildModel(json) {
  if (!json || typeof json !== 'object') {
    fail('模型 JSON 为空或格式错误');
  }
  if (json.format !== MODEL_FORMAT) {
    fail(
      `模型格式不匹配：期望 ${MODEL_FORMAT}，实际 ${json.format || '缺失'}。` +
        '旧版权重（baihe-multi-v3 / baihe-dualhead-v2 / baihe-charcnn-v1）已不再支持，请等待 Lead 导出 v4 五头权重。',
    );
  }
  const config = json.config;
  if (!config || typeof config !== 'object') {
    fail('模型缺少 config');
  }

  const maxLen = readInt(config.max_len, 'max_len', { min: 1 });
  const embDim = readInt(config.emb_dim, 'emb_dim', { min: 1 });
  const posDim = readInt(config.pos_dim, 'pos_dim', { min: 1 });
  const hidden = readInt(config.hidden, 'hidden', { min: 1 });
  const padId = readInt(config.pad_id, 'pad_id', { min: 0 });
  const unkId = readInt(config.unk_id, 'unk_id', { min: 0 });
  const bnEps = readNumber(config.bn_eps, 'bn_eps', { min: 0 });
  if (!(bnEps > 0)) fail(`模型 config.bn_eps 必须大于 0：${String(config.bn_eps)}`);

  const rawHeads = config.heads;
  if (!rawHeads || typeof rawHeads !== 'object') {
    fail('模型 config.heads 缺失');
  }
  for (const key of HEAD_KEYS) {
    if (!rawHeads[key]) fail(`模型 config.heads 缺少 ${key} 头`);
  }
  const headOrder = Object.keys(rawHeads).filter((key) => HEAD_KEYS.includes(key));
  for (const key of HEAD_KEYS) if (!headOrder.includes(key)) headOrder.push(key);

  if (!Array.isArray(json.vocab) || json.vocab.length < 2) {
    fail('模型 vocab 缺失或过短');
  }
  const vocab = json.vocab.slice();
  const vocabSize = vocab.length;
  if (padId >= vocabSize || unkId >= vocabSize) {
    fail('模型 config.pad_id / unk_id 超出 vocab 范围');
  }

  // 与 vocab.indexOf 语义一致：同一个 token 取首次出现的位置。
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

  const inputDim = embDim + posDim; // 每位置 40 维（concat，不是相加）

  const emb = readWeightBlock(weights.emb, [vocabSize, embDim], 'emb');
  const pos = readWeightBlock(weights.pos, [maxLen, posDim], 'pos');

  if (!Array.isArray(weights.conv) || weights.conv.length !== convSpec.length) {
    fail(`模型权重 conv 数量不符：期望 ${convSpec.length} 个分支`);
  }

  const conv = [];
  let baseDim = 0; // 池化后维度 = Σ 2*ch
  for (let i = 0; i < convSpec.length; i += 1) {
    const spec = convSpec[i] || {};
    const k = readInt(spec.k, `conv[${i}].k`, { min: 1 });
    const ch = readInt(spec.ch, `conv[${i}].ch`, { min: 1 });
    if (k > maxLen) {
      fail(`模型 config.conv[${i}].k=${k} 大于 max_len=${maxLen}`);
    }
    const branch = weights.conv[i];
    if (!branch || typeof branch !== 'object') {
      fail(`模型权重 conv[${i}] 缺失`);
    }
    conv.push({
      k,
      ch,
      W: readWeightBlock(branch.W, [ch, k * inputDim], `conv[${i}].W`),
      b: readWeightBlock(branch.b, [ch], `conv[${i}].b`),
      bn: readBnBlock(branch.bn, ch, `conv[${i}].bn`),
    });
    baseDim += 2 * ch;
  }

  const hiddenIn = baseDim + 2 * embDim; // 512 + 32 + 32 = 576
  const fcW = readWeightBlock(weights.fc && weights.fc.W, [hidden, hiddenIn], 'fc.W');
  const fcB = readWeightBlock(weights.fc && weights.fc.b, [hidden], 'fc.b');

  const rawHeadsWeights = weights.heads;
  if (!rawHeadsWeights || typeof rawHeadsWeights !== 'object') {
    fail('模型权重 heads 缺失');
  }
  const heads = {};
  for (const key of headOrder) {
    const headConfig = readHeadConfig(key, rawHeads[key]);
    const block = rawHeadsWeights[key];
    if (!block || typeof block !== 'object') fail(`模型权重 heads.${key} 缺失`);
    heads[key] = {
      ...headConfig,
      W: readWeightBlock(block.W, [headConfig.numClasses, hidden], `heads.${key}.W`),
      b: readWeightBlock(block.b, [headConfig.numClasses], `heads.${key}.b`),
    };
  }

  const yuri = heads.yuri;
  let yuriBaiheIndex = yuri.classes.indexOf('baihe');
  if (yuriBaiheIndex < 0) yuriBaiheIndex = 1;

  return {
    format: json.format,
    config: {
      max_len: maxLen,
      emb_dim: embDim,
      pos_dim: posDim,
      hidden,
      bn_eps: bnEps,
      pad_id: padId,
      unk_id: unkId,
      conv: convSpec.map((spec, i) => ({ k: conv[i].k, ch: conv[i].ch })),
      heads: Object.fromEntries(
        headOrder.map((key) => [key, { classes: heads[key].classes, names_zh: heads[key].namesZh }]),
      ),
    },
    vocab,
    vocabSize,
    charIndex,
    emb,
    pos,
    conv,
    hiddenIn,
    fcW,
    fcB,
    heads,
    headOrder,
    maxLen,
    embDim,
    posDim,
    inputDim,
    hidden,
    padId,
    unkId,
    bnEps,
    yuriBaiheIndex,
    metrics: json.metrics && typeof json.metrics === 'object' ? json.metrics : null,
    dataset: json.dataset && typeof json.dataset === 'object' ? json.dataset : null,
  };
}

/**
 * 读取模型 JSON 并构建模型（浏览器用）。
 * @param {string} url 模型地址，默认 ./model.json
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
 * 清洗输入（v4 规范）：
 *   1) 取字符串并 trim
 *   2) 去掉所有空白字符（\s）
 *   3) ASCII 大写字母（A-Z）统一转小写（与 train/ref_predict_v4.py 一致）
 *   4) 清洗后为空 → 抛「请输入名字」
 * 网名里的数字、下划线、点、连字符等原样保留，交给 vocab 查表。
 * 返回清洗后的字符串（不截断；截断在 predict 里按码点做）。
 */
export function normalizeInput(rawInput) {
  const text = rawInput == null ? '' : String(rawInput);
  const cleaned = text.trim().replace(/\s+/g, '').replace(/[A-Z]/g, (ch) => ch.toLowerCase());
  if (!cleaned) fail('请输入名字');
  return cleaned;
}

/** 稳定 softmax（减最大值）。 */
function softmax(logits) {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }
  const probs = new Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const value = Math.exp(logits[i] - maxLogit);
    probs[i] = value;
    sum += value;
  }
  for (let i = 0; i < logits.length; i += 1) probs[i] /= sum;
  return probs;
}

/**
 * 由字符 id 序列前向计算（内部函数）。
 * 输入 ids 长度 = max_len，已补 pad。
 */
function forwardFromIds(model, ids) {
  const { maxLen: L, embDim: E, posDim: P, inputDim: D, hidden: H, hiddenIn, padId, bnEps } = model;

  // 1) 每位置特征 X[i] = concat(emb[ids[i]], pos[i]) → 12 × 40
  const X = new Float64Array(L * D);
  for (let i = 0; i < L; i += 1) {
    const id = ids[i];
    if (!Number.isInteger(id) || id < 0 || id >= model.vocabSize) {
      fail(`字符 id 越界：${id}`);
    }
    const eBase = id * E;
    const xBase = i * D;
    for (let e = 0; e < E; e += 1) X[xBase + e] = model.emb[eBase + e];
    const pBase = i * P;
    for (let p = 0; p < P; p += 1) X[xBase + E + p] = model.pos[pBase + p];
  }

  // 2) 三路 conv → BatchNorm → ReLU → max/mean 双池化
  const pooled = new Float64Array(hiddenIn); // 前 512 维：每路 [max(ch), mean(ch)]；后 64 维：首/末字 emb
  let offset = 0;
  for (const branch of model.conv) {
    const { k, ch, W, b, bn } = branch;
    const windows = L - k + 1;
    for (let c = 0; c < ch; c += 1) {
      let maxVal = 0; // ReLU 后所有值 >= 0，从 0 起即可
      let sumVal = 0;
      const wBase = c * k * D;
      const scale = bn.gamma[c] / Math.sqrt(bn.var[c] + bnEps);
      const shift = bn.beta[c] - bn.mean[c] * scale;
      for (let p = 0; p < windows; p += 1) {
        let z = b[c];
        for (let j = 0; j < k; j += 1) {
          const xBase = (p + j) * D;
          const wOff = wBase + j * D;
          for (let e = 0; e < D; e += 1) z += W[wOff + e] * X[xBase + e];
        }
        const activated = z * scale + shift; // BatchNorm：(z - mean) / sqrt(var + eps) * gamma + beta
        const relu = activated > 0 ? activated : 0;
        if (relu > maxVal) maxVal = relu;
        sumVal += relu;
      }
      pooled[offset + c] = maxVal; // max_p ReLU(BN(z))
      pooled[offset + ch + c] = sumVal / windows; // mean_p ReLU(BN(z))
    }
    offset += 2 * ch;
  }

  // 3) 首字 emb + 末个非 pad 字 emb（各 32 维）
  let lastIndex = L - 1;
  while (lastIndex > 0 && ids[lastIndex] === padId) lastIndex -= 1;
  const firstBase = ids[0] * E;
  const lastBase = ids[lastIndex] * E;
  for (let e = 0; e < E; e += 1) {
    pooled[offset + e] = model.emb[firstBase + e];
    pooled[offset + E + e] = model.emb[lastBase + e];
  }

  // 4) fc: h = relu(fc.W · h0 + fc.b)，h0 = 576 维
  const h = new Float64Array(H);
  for (let o = 0; o < H; o += 1) {
    let z = model.fcB[o];
    const wBase = o * hiddenIn;
    for (let i = 0; i < hiddenIn; i += 1) z += model.fcW[wBase + i] * pooled[i];
    h[o] = z > 0 ? z : 0;
  }

  // 5) 四头各自 Linear + softmax
  const probs = {};
  for (const key of model.headOrder) {
    const head = model.heads[key];
    const out = new Float64Array(head.numClasses);
    for (let o = 0; o < head.numClasses; o += 1) {
      let z = head.b[o];
      const wBase = o * H;
      for (let i = 0; i < H; i += 1) z += head.W[wBase + i] * h[i];
      out[o] = z;
    }
    probs[key] = softmax(out);
  }

  return { probs };
}

/** 把输入清洗/编码成 ids（内部函数）。 */
function encodeInput(model, rawInput) {
  const cleaned = normalizeInput(rawInput);
  const codePoints = Array.from(cleaned);
  const maxLen = model.maxLen;
  const truncated = codePoints.length > maxLen;
  const chars = codePoints.slice(0, maxLen);

  const ids = new Array(maxLen).fill(model.padId);
  const unknownChars = [];
  for (let i = 0; i < chars.length; i += 1) {
    const char = chars[i];
    const id = model.charIndex.get(char);
    if (id === undefined) {
      ids[i] = model.unkId;
      unknownChars.push(char);
    } else {
      ids[i] = id;
    }
  }

  return { ids, chars, truncated, inputLength: codePoints.length, unknownChars };
}

/**
 * 预测一个名字的四个头概率。
 *
 * @param {object} model     buildModel / loadModel 得到的模型
 * @param {string} rawInput  原始输入（可含空白/大小写；中文、拉丁字母、混输均可）
 * @returns {{probs: {gender: number[], expression: number[], genre: number[], yuri: number[]},
 *            baiheIndex: number, chars: string[], ids: number[], truncated: boolean,
 *            inputLength: number, unknownChars: string[]}}
 *   probs.* 顺序与 config.heads.*.classes 一致；
 *   baiheIndex = probs.yuri[百合位] = P(百合)，即页面大号「百合指数」（0~1）。
 */
export function predict(model, rawInput) {
  if (!model || !model.emb) fail('模型未加载');
  const { ids, chars, truncated, inputLength, unknownChars } = encodeInput(model, rawInput);
  const { probs } = forwardFromIds(model, ids);
  return {
    probs,
    baiheIndex: probs.yuri[model.yuriBaiheIndex],
    chars,
    ids,
    truncated,
    inputLength,
    unknownChars,
  };
}

/**
 * 解析遮挡归因的目标类别：
 *   显式给 classIndex → 用它；
 *   否则 head='yuri' 用「百合」类；其它头用原输入下该头的 argmax 类。
 */
function resolveHeadTarget(model, probs, headKey, classIndex) {
  const head = model.heads[headKey];
  if (!head) fail(`未知的输出头：${headKey}（可选：${model.headOrder.join(' / ')}）`);
  const values = probs[headKey];
  let index;
  if (Number.isInteger(classIndex) && classIndex >= 0 && classIndex < head.numClasses) {
    index = classIndex;
  } else if (headKey === 'yuri') {
    index = model.yuriBaiheIndex;
  } else {
    index = 0;
    for (let i = 1; i < values.length; i += 1) if (values[i] > values[index]) index = i;
  }
  return { head, index, value: values[index] };
}

/**
 * 逐字遮挡归因：把第 i 个字符替换成 unk_id 后重新前向，返回指定头、指定类别概率的变化。
 *
 * delta = P(target | 原输入) − P(target | 该字换成 <unk>)
 *   delta > 0 → 这个字在给目标概率加分；delta < 0 → 在拖后腿
 *
 * @param {object} model
 * @param {string} rawInput
 * @param {string} [head='yuri']           'yuri' | 'gender' | 'expression' | 'genre'
 * @param {number|null} [classIndex=null]  目标类别下标；缺省见 resolveHeadTarget
 * @returns {Array<{index:number, char:string, delta:number, probWithout:number,
 *                  head:string, classIndex:number, className:string}>}
 */
export function occlusionAttribution(model, rawInput, head = 'yuri', classIndex = null) {
  const base = predict(model, rawInput);
  const { head: headDef, index, value: baseProb } = resolveHeadTarget(model, base.probs, head, classIndex);
  const unkId = model.unkId;

  return base.chars.map((char, position) => {
    const occludedIds = base.ids.slice();
    occludedIds[position] = unkId;
    const occluded = forwardFromIds(model, occludedIds);
    const probWithout = occluded.probs[head][index];
    return {
      index: position,
      char,
      delta: baseProb - probWithout,
      probWithout,
      head,
      classIndex: index,
      className: headDef.namesZh[index] || headDef.classes[index],
    };
  });
}

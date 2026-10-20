/**
 * 百合雷达 BAIHE RADAR · 前端推理引擎 v2（ES module）
 *
 * 严格复刻 `train/train_v2.py`（DualHead）导出的 baihe-dualhead-v2 权重：
 *   去空白 → 按 Unicode 码点取前 max_len 个字符 → vocab 查表（未登录用 unk_id）
 *   → 不足补 pad_id → X[i] = concat(emb[ids[i]] 32 维, pos[i] 8 维)（每位置 40 维）
 *   → 三路 conv(k=1/2/3, 行主序 W[c][j*40+e]) → BatchNorm → ReLU → 每路 max + mean 双池化
 *   → 按分支顺序拼 [max, mean] → 512 维 → 再拼首字 emb 与末个非 pad 字 emb（各 32）→ 576 维
 *   → fc(128) + ReLU → 双头：out3 → softmax 三分类；out2 → softmax 二分类（[非百合, 百合]）
 *
 * 纯娱乐向玩具；本文件只做推理，不含任何 UI。
 *
 * 对外接口：
 *   loadModel(url)                    → Promise<Model>（浏览器 fetch 模型 JSON）
 *   buildModel(json)                  → Model（同步构建，Node 对拍/测试用）
 *   predict(model, rawInput)          → {probs3, probs2, baiheIndex, chars, ids, truncated,
 *                                        inputLength, unknownChars}
 *   occlusionAttribution(model, raw)  → [{index, char, delta, indexWithout}, ...]
 *   normalizeInput(raw)               → string（去空白；空输入抛「请输入名字」）
 */

/** v2 权重格式标识（强制校验，避免误把旧 v1 权重当 v2 加载） */
export const MODEL_FORMAT = 'baihe-dualhead-v2';

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

/**
 * 由模型 JSON 构建可复用的推理模型对象（同步）。
 * @param {object} json  web/model.json 的内容（必须 format = baihe-dualhead-v2）
 * @returns {object}
 */
export function buildModel(json) {
  if (!json || typeof json !== 'object') {
    fail('模型 JSON 为空或格式错误');
  }
  if (json.format !== MODEL_FORMAT) {
    fail(
      `模型格式不匹配：期望 ${MODEL_FORMAT}，实际 ${json.format || '缺失'}。` +
        '旧版 v1 权重（baihe-charcnn-v1）已不再支持，请等待 Lead 导出 v2 双头权重。',
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
  const numClasses = readInt(config.num_classes, 'num_classes', { min: 1 });
  const numBinary = readInt(config.num_binary, 'num_binary', { min: 1 });
  const padId = readInt(config.pad_id, 'pad_id', { min: 0 });
  const unkId = readInt(config.unk_id, 'unk_id', { min: 0 });
  const bnEps = readNumber(config.bn_eps, 'bn_eps', { min: 0 });
  if (!(bnEps > 0)) fail(`模型 config.bn_eps 必须大于 0：${String(config.bn_eps)}`);

  const classes = Array.isArray(config.classes) ? config.classes.slice() : [];
  if (classes.length !== numClasses) {
    fail(`模型 config.classes 数量不符：期望 ${numClasses} 个，实际 ${classes.length} 个`);
  }
  const classNamesZh =
    Array.isArray(config.class_names_zh) && config.class_names_zh.length === numClasses
      ? config.class_names_zh.slice()
      : classes.slice();

  const binaryLabels = Array.isArray(config.binary_labels) ? config.binary_labels.slice() : [];
  if (binaryLabels.length !== numBinary) {
    fail(`模型 config.binary_labels 数量不符：期望 ${numBinary} 个，实际 ${binaryLabels.length} 个`);
  }
  const binaryBaiheIndex = binaryLabels.indexOf('baihe');
  if (binaryBaiheIndex < 0) {
    fail('模型 config.binary_labels 缺少 baihe，无法计算百合指数');
  }
  let baiheIndex3 = classes.indexOf('baihe');
  if (baiheIndex3 < 0) baiheIndex3 = 0;

  const feat = Array.isArray(config.feat) ? config.feat.slice() : ['first_last_emb'];
  if (!feat.includes('first_last_emb')) {
    fail(`模型 config.feat 缺少 first_last_emb：${JSON.stringify(config.feat)}`);
  }

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
  const out3W = readWeightBlock(weights.out3 && weights.out3.W, [numClasses, hidden], 'out3.W');
  const out3B = readWeightBlock(weights.out3 && weights.out3.b, [numClasses], 'out3.b');
  const out2W = readWeightBlock(weights.out2 && weights.out2.W, [numBinary, hidden], 'out2.W');
  const out2B = readWeightBlock(weights.out2 && weights.out2.b, [numBinary], 'out2.b');

  return {
    format: json.format,
    config: {
      max_len: maxLen,
      emb_dim: embDim,
      pos_dim: posDim,
      hidden,
      num_classes: numClasses,
      num_binary: numBinary,
      pad_id: padId,
      unk_id: unkId,
      bn_eps: bnEps,
      conv: convSpec.map((spec, i) => ({ k: conv[i].k, ch: conv[i].ch })),
      classes,
      class_names_zh: classNamesZh,
      binary_labels: binaryLabels,
      feat,
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
    out3W,
    out3B,
    out2W,
    out2B,
    maxLen,
    embDim,
    posDim,
    inputDim,
    hidden,
    numClasses,
    numBinary,
    padId,
    unkId,
    bnEps,
    baiheIndex3,
    binaryBaiheIndex,
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
 * 清洗输入：取字符串、去掉所有空白字符（\s）；清洗后为空 → 抛「请输入名字」。
 */
export function normalizeInput(rawInput) {
  const text = rawInput == null ? '' : String(rawInput);
  const cleaned = text.replace(/\s+/g, '');
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
  const {
    maxLen: L,
    embDim: E,
    posDim: P,
    inputDim: D,
    hidden: H,
    hiddenIn,
    padId,
    bnEps,
  } = model;

  // 1) 每位置特征 X[i] = concat(emb[ids[i]], pos[i]) → 8 × 40
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
  const firstId = ids[0];
  const lastId = ids[lastIndex];
  const firstBase = firstId * E;
  const lastBase = lastId * E;
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

  // 5) 双头
  const logits3 = new Float64Array(model.numClasses);
  for (let o = 0; o < model.numClasses; o += 1) {
    let z = model.out3B[o];
    const wBase = o * H;
    for (let i = 0; i < H; i += 1) z += model.out3W[wBase + i] * h[i];
    logits3[o] = z;
  }
  const logits2 = new Float64Array(model.numBinary);
  for (let o = 0; o < model.numBinary; o += 1) {
    let z = model.out2B[o];
    const wBase = o * H;
    for (let i = 0; i < H; i += 1) z += model.out2W[wBase + i] * h[i];
    logits2[o] = z;
  }

  return {
    probs3: softmax(logits3),
    probs2: softmax(logits2),
    logits3: Array.from(logits3),
    logits2: Array.from(logits2),
  };
}

/** 把输入清洗/编码成 ids（内部函数）。 */
function encodeInput(model, rawInput) {
  const cleaned = normalizeInput(rawInput);
  const codePoints = Array.from(cleaned);
  const maxLen = model.maxLen;
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

  return { ids, chars, truncated, inputLength: codePoints.length, unknownChars };
}

/**
 * 预测一个名字。
 *
 * @param {object} model     buildModel / loadModel 得到的模型
 * @param {string} rawInput  原始输入（可含空白，可超长）
 * @returns {{probs3: number[], probs2: number[], baiheIndex: number, chars: string[],
 *            ids: number[], truncated: boolean, inputLength: number, unknownChars: string[]}}
 *   probs3  三分类概率，顺序 = config.classes（契约固定 [baihe, purelove, romance]）
 *   probs2  二分类概率，顺序 = config.binary_labels（契约固定 [not_baihe, baihe]）
 *   baiheIndex = probs2[百合位] = P(百合)，即页面大号「百合指数」（0~1）
 */
export function predict(model, rawInput) {
  if (!model || !model.emb) fail('模型未加载');
  const { ids, chars, truncated, inputLength, unknownChars } = encodeInput(model, rawInput);
  const { probs3, probs2 } = forwardFromIds(model, ids);
  return {
    probs3,
    probs2,
    baiheIndex: probs2[model.binaryBaiheIndex],
    chars,
    ids,
    truncated,
    inputLength,
    unknownChars,
  };
}

/**
 * 逐字遮挡归因：把第 i 个字符替换成 unk_id 后重新前向，返回「百合指数」的变化。
 *
 * delta = P(百合 | 原输入) − P(百合 | 该字换成 <unk>)
 *   delta > 0 → 这个字在给百合指数加分；delta < 0 → 在拖后腿
 *
 * @param {object} model
 * @param {string} rawInput
 * @returns {Array<{index:number, char:string, delta:number, indexWithout:number}>}
 */
export function occlusionAttribution(model, rawInput) {
  const base = predict(model, rawInput);
  const unkId = model.config.unk_id;
  const baseIndex = base.baiheIndex;

  return base.chars.map((char, index) => {
    const occludedIds = base.ids.slice();
    occludedIds[index] = unkId;
    const occluded = forwardFromIds(model, occludedIds);
    const indexWithout = occluded.probs2[model.binaryBaiheIndex];
    return {
      index,
      char,
      delta: baseIndex - indexWithout,
      indexWithout,
    };
  });
}

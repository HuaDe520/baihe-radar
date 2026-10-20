#!/usr/bin/env node
/**
 * 生成 v2 双头模型的调试用随机权重：web/mock-model-v2.json
 *
 * 输出严格符合 baihe-dualhead-v2 契约：
 *   config.max_len=8, emb_dim=32, pos_dim=8, conv k=1/2/3 → 64/96/96,
 *   hidden=128, num_classes=3, num_binary=2, bn_eps=1e-5,
 *   fc [128, 576], out3 [3,128], out2 [2,128]
 * 权重为固定种子随机初始化（数值保留 6 位小数），仅用于前端调试 / JS 自测，不是训练结果。
 *
 * 用法：
 *   node web/make_mock_model_v2.mjs                     # 写 web/mock-model-v2.json
 *   node web/make_mock_model_v2.mjs path/to/out.json    # 自定义输出
 */

import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildModel, predict } from './inference.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));

// ---------- 固定种子随机数 ----------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function rand() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(0x0ba1ae2);
function randn() {
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function randomArray(length, scale) {
  const out = new Array(length);
  for (let i = 0; i < length; i += 1) out[i] = randn() * scale;
  return out;
}
const round6 = (value) => Math.round(value * 1e6) / 1e6;
const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));

// ---------- 词表：<pad> + <unk> + 常见名字用字（V=120） ----------
const CHIP_CHARS = '林晚苏清沈月顾言张伟王芳李娜陈星';
const EXTRA_CHARS =
  '安然若雨晴夏北南初栀念白杨梦瑶雪琳静怡心语思柔宁沐晨曦云风花叶秋冬春江海山川光影霜灵芷罗兰竹梅菊香韵诗词歌赋琴棋书画剑武湖魔道仙侠都市校园青古代近未来际末世异能热血治愈悬疑推理奇幻神话明辰夜雅婉芸妍姝婷薇蕾蕊莲荷茉莉菲雯琦琪瑾瑜';
const VOCAB_TARGET = 120;

const uniqueChars = [...new Set(Array.from(CHIP_CHARS + EXTRA_CHARS))];
if (uniqueChars.length < VOCAB_TARGET - 2) {
  throw new Error(`mock 用字池不足：只有 ${uniqueChars.length} 个字，需要 ${VOCAB_TARGET - 2} 个`);
}
const vocab = ['<pad>', '<unk>', ...uniqueChars.slice(0, VOCAB_TARGET - 2)];
const V = vocab.length;

// ---------- 结构超参（与 train_v2.py / 契约一致） ----------
const MAX_LEN = 8;
const EMB_DIM = 32;
const POS_DIM = 8;
const HIDDEN = 128;
const NUM_CLASSES = 3;
const NUM_BINARY = 2;
const BN_EPS = 1e-5;
const CONV_SPEC = [
  { k: 1, ch: 64 },
  { k: 2, ch: 96 },
  { k: 3, ch: 96 },
];
const INPUT_DIM = EMB_DIM + POS_DIM; // 40
const BASE_DIM = CONV_SPEC.reduce((sum, item) => sum + 2 * item.ch, 0); // 512
const HIDDEN_IN = BASE_DIM + 2 * EMB_DIM; // 576

// ---------- 随机权重 ----------
const embData = randomArray(V * EMB_DIM, 0.38);
for (let e = 0; e < EMB_DIM; e += 1) embData[e] = 0; // pad 行置零，贴近真实 padding_idx
const posData = randomArray(MAX_LEN * POS_DIM, 0.22);

const convWeights = CONV_SPEC.map(({ k, ch }) => {
  const scale = Math.sqrt(2 / (k * INPUT_DIM));
  const gamma = new Array(ch);
  const beta = new Array(ch);
  const mean = new Array(ch);
  const variance = new Array(ch);
  for (let c = 0; c < ch; c += 1) {
    gamma[c] = clamp(1 + randn() * 0.08, 0.5, 1.5);
    beta[c] = randn() * 0.05;
    mean[c] = randn() * 0.05;
    variance[c] = clamp(1 + randn() * 0.12, 0.3, 3);
  }
  return {
    k,
    W: { shape: [ch, k * INPUT_DIM], data: randomArray(ch * k * INPUT_DIM, scale) },
    b: { shape: [ch], data: new Array(ch).fill(0) },
    bn: {
      gamma: { shape: [ch], data: gamma },
      beta: { shape: [ch], data: beta },
      mean: { shape: [ch], data: mean },
      var: { shape: [ch], data: variance },
    },
  };
});

const fcScale = Math.sqrt(2 / HIDDEN_IN);
const out3Scale = Math.sqrt(2 / HIDDEN) * 1.6;
const out2Scale = Math.sqrt(2 / HIDDEN) * 1.3;

const model = {
  format: 'baihe-dualhead-v2',
  config: {
    max_len: MAX_LEN,
    emb_dim: EMB_DIM,
    pos_dim: POS_DIM,
    conv: CONV_SPEC.map(({ k, ch }) => ({ k, ch })),
    hidden: HIDDEN,
    num_classes: NUM_CLASSES,
    num_binary: NUM_BINARY,
    pad_id: 0,
    unk_id: 1,
    bn_eps: BN_EPS,
    classes: ['baihe', 'purelove', 'romance'],
    class_names_zh: ['百合', '纯爱(BL)', '言情(BG)'],
    binary_labels: ['not_baihe', 'baihe'],
    feat: ['first_last_emb'],
  },
  vocab,
  weights: {
    emb: { shape: [V, EMB_DIM], data: embData },
    pos: { shape: [MAX_LEN, POS_DIM], data: posData },
    conv: convWeights,
    fc: {
      W: { shape: [HIDDEN, HIDDEN_IN], data: randomArray(HIDDEN * HIDDEN_IN, fcScale) },
      b: { shape: [HIDDEN], data: new Array(HIDDEN).fill(0) },
    },
    out3: {
      W: { shape: [NUM_CLASSES, HIDDEN], data: randomArray(NUM_CLASSES * HIDDEN, out3Scale) },
      b: { shape: [NUM_CLASSES], data: new Array(NUM_CLASSES).fill(0) },
    },
    out2: {
      W: { shape: [NUM_BINARY, HIDDEN], data: randomArray(NUM_BINARY * HIDDEN, out2Scale) },
      b: { shape: [NUM_BINARY], data: new Array(NUM_BINARY).fill(0) },
    },
  },
  // mock 没有训练指标：n_train = 0 → 前端模型卡显示「待训练」
  metrics: {
    test_acc_3cls: 0,
    baseline_3cls: 0.333,
    macro_f1_3cls: 0,
    binary_acc: 0,
    binary_balanced_acc: 0,
    binary_macro_f1: 0,
    per_class: {
      baihe: { precision: 0, recall: 0, f1: 0 },
      purelove: { precision: 0, recall: 0, f1: 0 },
      romance: { precision: 0, recall: 0, f1: 0 },
    },
    confusion: [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ],
    binary_confusion: [
      [0, 0],
      [0, 0],
    ],
    n_train: 0,
    n_val: 0,
    n_test: 0,
    params: 0,
    best_epoch: 0,
  },
  dataset: {
    baihe: 0,
    purelove: 0,
    romance: 0,
    source: '晋江文学城(jjwxc.net)作品库 主角名',
  },
  meta: {
    note: 'MOCK WEIGHTS v2 · 随机初始化，仅用于前端调试，不是训练结果',
    generated_at: new Date().toISOString(),
    seed: '0x0ba1ae2',
  },
};

// 参数数量（约等于真实模型参数量）
const bnParams = CONV_SPEC.reduce((sum, item) => sum + 4 * item.ch, 0);
model.metrics.params =
  V * EMB_DIM + MAX_LEN * POS_DIM +
  CONV_SPEC.reduce((sum, item) => sum + item.ch * (item.k * INPUT_DIM + 1), 0) +
  bnParams + HIDDEN * HIDDEN_IN + HIDDEN + NUM_CLASSES * HIDDEN + NUM_CLASSES +
  NUM_BINARY * HIDDEN + NUM_BINARY;

// ---------- 数值保留 6 位小数（契约要求） ----------
function roundWeightsInPlace(node) {
  if (!node || typeof node !== 'object') return;
  if (Array.isArray(node.data)) node.data = node.data.map(round6);
  for (const key of Object.keys(node)) {
    if (key === 'data') continue;
    const value = node[key];
    if (Array.isArray(value)) value.forEach(roundWeightsInPlace);
    else if (value && typeof value === 'object') roundWeightsInPlace(value);
  }
}
roundWeightsInPlace(model.weights);

// ---------- 写文件 ----------
const outPath = path.resolve(process.argv[2] || path.join(HERE, 'mock-model-v2.json'));
const json = JSON.stringify(model);
writeFileSync(outPath, `${json}\n`, 'utf8');

// ---------- 自检：用刚写出的权重跑一遍前端推理引擎 ----------
const built = buildModel(JSON.parse(json));
const demos = ['林晚', '苏清', '沈月', '王芳', '未知字阿', '这是一个超过八个字的测试名字'];
const lines = demos.map((name) => {
  const { probs3, probs2, baiheIndex, truncated, unknownChars } = predict(built, name);
  const flags = [];
  if (truncated) flags.push('截断');
  if (unknownChars.length) flags.push(`未登录字 ${unknownChars.length}`);
  return (
    `  ${name}  →  百合指数 ${(baiheIndex * 100).toFixed(1)}%  |  ` +
    `三类 ${probs3.map((p) => `${(p * 100).toFixed(1)}%`).join(' / ')}  |  ` +
    `二类 [${probs2.map((p) => `${(p * 100).toFixed(1)}%`).join(', ')}]` +
    (flags.length ? `  (${flags.join('，')})` : '')
  );
});

console.log(`✔ 已生成 ${outPath}`);
console.log(
  `  词表 V=${V}，emb=${V}×${EMB_DIM} + pos=${MAX_LEN}×${POS_DIM} → 每位置 ${INPUT_DIM} 维；` +
    `conv=${CONV_SPEC.map((c) => `k${c.k}:${c.ch}`).join(' / ')} + BN；池化 ${BASE_DIM} + 首末字 ${2 * EMB_DIM} = ${HIDDEN_IN}；fc=${HIDDEN}；双头 ${NUM_CLASSES}/${NUM_BINARY}`,
);
console.log(`  文件大小 ${(Buffer.byteLength(json) / 1024).toFixed(1)} KB，权重保留 6 位小数，参数量约 ${model.metrics.params}`);
console.log('  示例预测（随机权重，仅供调试）：');
console.log(lines.join('\n'));
console.log('  提示：临时调试可复制为 web/model.json；正式 web/model.json 由 Lead 导出覆盖，本脚本不要动它。');

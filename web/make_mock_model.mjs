#!/usr/bin/env node
/**
 * 生成调试用的随机权重模型：web/mock-model.json
 *
 * 输出严格符合 CONTRACT.md 第二节的 JSON 形状（V≈120，行主序展平、数值保留 6 位小数），
 * 但权重是固定种子随机初始化的 —— 只用来本地调试前端 / 自测 JS 推理，不是训练结果。
 *
 * 用法：
 *   node web/make_mock_model.mjs                 # 写 web/mock-model.json
 *   node web/make_mock_model.mjs path/to/out.json
 */

import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildModel, predict } from './inference.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));

// ---------- 固定种子的随机数（结果可复现） ----------
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
const rand = mulberry32(0x0ba1ae1); // 固定种子
function randn() {
  // Box–Muller，均值 0、方差 1
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

// ---------- 词表：<pad> + <unk> + 常见名字用字（V≈120） ----------
const CHIP_CHARS = '林晚苏清沈月顾言张伟王芳李娜陈星';
const EXTRA_CHARS =
  '安然若雨晴夏北南初栀念白杨梦瑶雪琳静怡心语思柔宁沐晨曦云风花叶秋冬春江海山川光影霜灵芷罗兰竹梅菊香韵诗词歌赋琴棋书画剑武湖魔道仙侠都市校园青古代近未来际末世异能热血治愈悬疑推理奇幻神话明辰夜雅婉芸妍姝婷薇蕾蕊莲荷茉莉菲雯琦琪瑾瑜';
const VOCAB_TARGET = 120; // 契约示例：V 约 120

const uniqueChars = [...new Set(Array.from(CHIP_CHARS + EXTRA_CHARS))];
if (uniqueChars.length < VOCAB_TARGET - 2) {
  throw new Error(`mock 用字池不足：只有 ${uniqueChars.length} 个字，需要 ${VOCAB_TARGET - 2} 个`);
}
const vocab = ['<pad>', '<unk>', ...uniqueChars.slice(0, VOCAB_TARGET - 2)];
const V = vocab.length;

// ---------- 结构超参（与契约一致） ----------
const MAX_LEN = 8;
const EMB_DIM = 28;
const HIDDEN = 64;
const NUM_CLASSES = 3;
const CONV_SPEC = [
  { k: 1, ch: 32 },
  { k: 2, ch: 40 },
  { k: 3, ch: 40 },
];
const HIDDEN_IN = CONV_SPEC.reduce((sum, item) => sum + item.ch, 0); // 112

// ---------- 随机权重（Kaiming 风格缩放，模拟训练后的量级） ----------
const embData = randomArray(V * EMB_DIM, 0.38);
// 把 pad 行的 embedding 设为 0，更像真实训练里的 padding 0 向量
for (let e = 0; e < EMB_DIM; e += 1) embData[e] = 0;

const convWeights = CONV_SPEC.map(({ k, ch }) => {
  const fanIn = k * EMB_DIM;
  const scale = Math.sqrt(2 / fanIn);
  return {
    k,
    W: { shape: [ch, k * EMB_DIM], data: randomArray(ch * k * EMB_DIM, scale) },
    b: { shape: [ch], data: new Array(ch).fill(0) },
  };
});

const fcScale = Math.sqrt(2 / HIDDEN_IN);
const outScale = Math.sqrt(2 / HIDDEN) * 1.8; // 稍微放大，让 mock 输出更有区分度

const model = {
  format: 'baihe-charcnn-v1',
  config: {
    max_len: MAX_LEN,
    emb_dim: EMB_DIM,
    hidden: HIDDEN,
    num_classes: NUM_CLASSES,
    pad_id: 0,
    unk_id: 1,
    conv: CONV_SPEC.map(({ k, ch }) => ({ k, ch })),
    classes: ['baihe', 'purelove', 'romance'],
    class_names_zh: ['百合', '纯爱(BL)', '言情(BG)'],
  },
  vocab,
  weights: {
    emb: { shape: [V, EMB_DIM], data: embData },
    conv: convWeights,
    fc: {
      W: { shape: [HIDDEN, HIDDEN_IN], data: randomArray(HIDDEN * HIDDEN_IN, fcScale) },
      b: { shape: [HIDDEN], data: new Array(HIDDEN).fill(0) },
    },
    out: {
      W: { shape: [NUM_CLASSES, HIDDEN], data: randomArray(NUM_CLASSES * HIDDEN, outScale) },
      b: { shape: [NUM_CLASSES], data: new Array(NUM_CLASSES).fill(0) },
    },
  },
  // mock 没有训练指标：n_train = 0 → 前端模型卡会显示「待训练」
  metrics: {
    test_acc: 0,
    baseline: 0.333,
    macro_f1: 0,
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
    n_train: 0,
    n_test: 0,
  },
  dataset: {
    baihe: 0,
    purelove: 0,
    romance: 0,
    source: '晋江文学城(jjwxc.net)作品库 主角名',
  },
  meta: {
    note: 'MOCK WEIGHTS · 随机初始化，仅用于前端调试，不是训练结果',
    generated_at: new Date().toISOString(),
    seed: '0x0ba1ae1',
  },
};

// 数值四舍五入保留 6 位小数（契约要求）
function roundWeightsInPlace(node) {
  if (!node || typeof node !== 'object') return;
  if (Array.isArray(node.data)) {
    node.data = node.data.map(round6);
  }
  for (const key of Object.keys(node)) {
    if (key === 'data') continue;
    const value = node[key];
    if (Array.isArray(value)) value.forEach(roundWeightsInPlace);
    else if (value && typeof value === 'object') roundWeightsInPlace(value);
  }
}
roundWeightsInPlace(model.weights);

// ---------- 写文件 ----------
const outPath = path.resolve(process.argv[2] || path.join(HERE, 'mock-model.json'));
const json = JSON.stringify(model);
writeFileSync(outPath, `${json}\n`, 'utf8');

// ---------- 自检：用刚写出的权重跑一遍前端推理引擎 ----------
const built = buildModel(JSON.parse(json));
const demos = ['林晚', '苏清', '沈月', '王芳', '未知字阿', '这是一个超过八个字的测试名字'];
const lines = demos.map((name) => {
  const { probs, truncated, unknownChars } = predict(built, name);
  const pct = probs.map((p) => `${(p * 100).toFixed(1)}%`).join(' / ');
  const flags = [];
  if (truncated) flags.push('截断');
  if (unknownChars.length) flags.push(`未登录字 ${unknownChars.length}`);
  return `  ${name}  →  ${pct}${flags.length ? `  (${flags.join('，')})` : ''}`;
});

console.log(`✔ 已生成 ${outPath}`);
console.log(`  词表 V=${V}，emb=${V}×${EMB_DIM}，conv=${CONV_SPEC.map((c) => `k${c.k}:${c.ch}`).join(' / ')}，fc=${HIDDEN}，out=${NUM_CLASSES}`);
console.log(`  文件大小 ${(Buffer.byteLength(json) / 1024).toFixed(1)} KB，权重已保留 6 位小数`);
console.log('  示例预测（随机权重，仅供调试）：');
console.log(lines.join('\n'));
if (!outPath.endsWith('model.json')) {
  console.log('  提示：临时调试可复制为 web/model.json；正式 model.json 由 Lead 导出覆盖，交付时前端会读取它。');
}

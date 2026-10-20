#!/usr/bin/env node
/**
 * v3 多头模型对拍脚本：JS 推理（web/inference.js） vs Python 参考实现。
 *
 * 用法（在任意目录执行均可）：
 *   node web/parity_test.mjs
 *       读 web/model.json + train/test_vectors3.json，
 *       逐条比对 4 个头（gender / expression / genre / yuri）概率的 max|Δ|。
 *   node web/parity_test.mjs --mock
 *       用 web/mock-model-v3.json 代替 web/model.json（本地无真模型时自测链路）。
 *   BAIHE_VECTORS=/path/to/vectors3.json node web/parity_test.mjs
 *       指定另一份 test vectors（临时对拍用）。
 *
 * 通过标准：4 个头全部 max|Δ| < 5e-4，退出码 0；否则 1。
 * 缺文件时打印提示并退出码 0：
 *   - web/model.json 不存在 → 提示等 Lead 导出 v3；
 *   - web/model.json 还是旧格式（非 baihe-multi-v3）→ 提示等 Lead 覆盖；
 *   - train/test_vectors3.json 不存在 → 打印「待 Lead 生成」。
 */

import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildModel, predict, MODEL_FORMAT, HEAD_KEYS } from './inference.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const TOLERANCE = 5e-4;
const VECTORS_FORMAT = 'baihe-multi-v3';

const useMock = process.argv.includes('--mock');
const modelPath = useMock ? path.join(HERE, 'mock-model-v3.json') : path.join(HERE, 'model.json');
const vectorsPath = process.env.BAIHE_VECTORS
  ? path.resolve(process.env.BAIHE_VECTORS)
  : path.join(ROOT, 'train', 'test_vectors3.json');

function readJson(filePath, label) {
  try {
    return JSON.parse(readFileSync(filePath, 'utf8'));
  } catch (err) {
    console.error(`✘ ${label} 解析失败：${filePath}`);
    console.error(`  ${err && err.message ? err.message : err}`);
    process.exit(1);
  }
}

function maxDiff(actual, expected) {
  let diff = 0;
  for (let i = 0; i < expected.length; i += 1) {
    diff = Math.max(diff, Math.abs(actual[i] - expected[i]));
  }
  return diff;
}

console.log('百合雷达 v3 · JS ↔ Python 推理对拍（性别 / ACG 属性 / 作品倾向 / 百合判定 四头）');
console.log(`  模型：${modelPath}`);
console.log(`  向量：${vectorsPath}`);

if (!existsSync(modelPath)) {
  console.log(`⏳ 未找到模型文件：${modelPath}`);
  console.log('   web/model.json 待 Lead 训练 v3 导出；本地可先 `node web/make_mock_model_v3.mjs` 生成 mock 权重，');
  console.log('   再用 `node web/parity_test.mjs --mock` 自测链路。跳过对拍，退出码 0。');
  process.exit(0);
}

const modelJson = readJson(modelPath, '模型 JSON');
if (modelJson.format !== MODEL_FORMAT) {
  console.log(`⏳ web/model.json 仍是旧格式「${modelJson.format || '缺失'}」，待 Lead 导出 ${MODEL_FORMAT}。`);
  console.log('   跳过对拍，退出码 0。');
  process.exit(0);
}

if (!existsSync(vectorsPath)) {
  console.log('⏳ train/test_vectors3.json 待 Lead 生成，跳过对拍（退出码 0）。');
  console.log('   生成后重跑本脚本即可：node web/parity_test.mjs');
  process.exit(0);
}

const vectors = readJson(vectorsPath, 'test_vectors3.json');
if (vectors.format && vectors.format !== VECTORS_FORMAT) {
  console.log(`⚠️  vectors 的 format 是「${vectors.format}」，不是 ${VECTORS_FORMAT}，继续按 v3 字段比对。`);
}
if (!vectors || !Array.isArray(vectors.cases) || vectors.cases.length === 0) {
  console.log('⏳ test_vectors3.json 里没有 cases，跳过对拍（退出码 0）。');
  process.exit(0);
}

const model = buildModel(modelJson);
const headKeys = model.headOrder.length ? model.headOrder : HEAD_KEYS;
const details = [];
const maxDiffs = Object.fromEntries(headKeys.map((key) => [key, 0]));
const worst = Object.fromEntries(headKeys.map((key) => [key, null]));
let failures = 0;

for (let i = 0; i < vectors.cases.length; i += 1) {
  const testCase = vectors.cases[i] || {};
  const input = typeof testCase.input === 'string' ? testCase.input : String(testCase.input ?? '');
  const expectedProbs = testCase.probs && typeof testCase.probs === 'object' ? testCase.probs : null;
  if (!expectedProbs) {
    failures += 1;
    details.push(`  ✘ #${i} ${JSON.stringify(input)} 缺少 probs 对象`);
    continue;
  }
  let shapeBad = false;
  for (const key of headKeys) {
    const expected = Array.isArray(expectedProbs[key]) ? expectedProbs[key].map(Number) : [];
    if (expected.length !== model.heads[key].numClasses) {
      failures += 1;
      shapeBad = true;
      details.push(
        `  ✘ #${i} ${JSON.stringify(input)} probs.${key} 长度不符：期望 ${model.heads[key].numClasses}，实际 ${expected.length}`,
      );
    }
  }
  if (shapeBad) continue;

  let got;
  try {
    got = predict(model, input);
  } catch (err) {
    failures += 1;
    details.push(`  ✘ #${i} ${JSON.stringify(input)} 前端推理抛错：${err && err.message ? err.message : err}`);
    continue;
  }

  const caseDiffs = {};
  let caseFail = false;
  for (const key of headKeys) {
    const diff = maxDiff(got.probs[key], expectedProbs[key].map(Number));
    caseDiffs[key] = diff;
    if (diff > maxDiffs[key]) {
      maxDiffs[key] = diff;
      worst[key] = { input, got: got.probs[key], expected: expectedProbs[key].map(Number) };
    }
    if (!(diff < TOLERANCE)) caseFail = true;
  }
  if (caseFail) {
    failures += 1;
    const parts = headKeys.map((key) => `Δ${key}=${caseDiffs[key].toExponential(3)}`);
    const rows = headKeys.map(
      (key) =>
        `     js.${key}=[${got.probs[key].map((v) => v.toFixed(6)).join(', ')}]  ` +
        `py.${key}=[${expectedProbs[key].map((v) => Number(v).toFixed(6)).join(', ')}]`,
    );
    details.push(`  ✘ #${i} ${JSON.stringify(input)} ${parts.join(' ')}\n${rows.join('\n')}`);
  }
}

console.log('');
console.log(`  用例数：${vectors.cases.length}`);
for (const key of headKeys) {
  const names = model.heads[key].namesZh;
  console.log(
    `  max|Δprobs.${key}|（${model.heads[key].numClasses} 类：${names.join('/')}）：` +
      `${maxDiffs[key].toExponential(3)}（阈值 ${TOLERANCE}）`,
  );
}
for (const key of headKeys) {
  if (!worst[key]) continue;
  console.log(
    `  最差 probs.${key}：${JSON.stringify(worst[key].input)}  ` +
      `js=[${worst[key].got.map((v) => v.toFixed(6)).join(', ')}]  ` +
      `py=[${worst[key].expected.map((v) => v.toFixed(6)).join(', ')}]`,
  );
}
if (details.length) {
  console.log('');
  console.log('  失败明细：');
  console.log(details.slice(0, 20).join('\n'));
  if (details.length > 20) console.log(`  …… 其余 ${details.length - 20} 条省略`);
}

if (failures === 0) {
  console.log(
    `✔ PASS：${vectors.cases.length} 条全部满足 4 头 max|Δ| < ${TOLERANCE}（格式 ${model.format}，` +
      `max_len=${model.maxLen}）`,
  );
  process.exit(0);
}
console.log(`✘ FAIL：${failures}/${vectors.cases.length} 条超出容差`);
process.exit(1);

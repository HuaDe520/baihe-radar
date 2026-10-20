#!/usr/bin/env node
/**
 * v2 双头模型对拍脚本：JS 推理（web/inference.js） vs Python 参考实现。
 *
 * 用法（在任意目录执行均可）：
 *   node web/parity_test.mjs
 *       读 web/model.json + train/test_vectors2.json，
 *       逐条比对 probs3（三分类）与 probs2（二分类）的 max|Δ|。
 *   node web/parity_test.mjs --mock
 *       用 web/mock-model-v2.json 代替 web/model.json（本地无真模型时自测链路）。
 *   BAIHE_VECTORS=/path/to/vectors2.json node web/parity_test.mjs
 *       指定另一份 test vectors（临时对拍用）。
 *
 * 通过标准：max|Δprobs3| < 5e-4 且 max|Δprobs2| < 5e-4，退出码 0；否则 1。
 * 缺文件时打印提示并退出码 0：
 *   - web/model.json 不存在 → 提示等 Lead 导出 v2；
 *   - web/model.json 还是旧格式（非 baihe-dualhead-v2）→ 提示等 Lead 覆盖；
 *   - train/test_vectors2.json 不存在 → 打印「待 Lead 生成」。
 */

import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildModel, predict, MODEL_FORMAT } from './inference.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const TOLERANCE = 5e-4;
const VECTORS_FORMAT = 'baihe-dualhead-v2';

const useMock = process.argv.includes('--mock');
const modelPath = useMock ? path.join(HERE, 'mock-model-v2.json') : path.join(HERE, 'model.json');
const vectorsPath = process.env.BAIHE_VECTORS
  ? path.resolve(process.env.BAIHE_VECTORS)
  : path.join(ROOT, 'train', 'test_vectors2.json');

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

console.log('百合雷达 v2 · JS ↔ Python 推理对拍（三分类 + 二分类双头）');
console.log(`  模型：${modelPath}`);
console.log(`  向量：${vectorsPath}`);

if (!existsSync(modelPath)) {
  console.log(`⏳ 未找到模型文件：${modelPath}`);
  console.log('   web/model.json 待 Lead 训练 v2 导出；本地可先 `node web/make_mock_model_v2.mjs` 生成 mock 权重，');
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
  console.log('⏳ train/test_vectors2.json 待 Lead 生成，跳过对拍（退出码 0）。');
  console.log('   生成后重跑本脚本即可：node web/parity_test.mjs');
  process.exit(0);
}

const vectors = readJson(vectorsPath, 'test_vectors2.json');
if (vectors.format && vectors.format !== VECTORS_FORMAT) {
  console.log(`⚠️  vectors 的 format 是「${vectors.format}」，不是 ${VECTORS_FORMAT}，继续按 v2 字段比对。`);
}
if (!vectors || !Array.isArray(vectors.cases) || vectors.cases.length === 0) {
  console.log('⏳ test_vectors2.json 里没有 cases，跳过对拍（退出码 0）。');
  process.exit(0);
}

const model = buildModel(modelJson);
const classes = model.config.classes;
const binaryLabels = model.config.binary_labels;
const details = [];
let maxDiff3 = 0;
let maxDiff2 = 0;
let worst3 = null;
let worst2 = null;
let failures = 0;

for (let i = 0; i < vectors.cases.length; i += 1) {
  const testCase = vectors.cases[i] || {};
  const input = typeof testCase.input === 'string' ? testCase.input : String(testCase.input ?? '');
  const expected3 = Array.isArray(testCase.probs3) ? testCase.probs3.map(Number) : [];
  const expected2 = Array.isArray(testCase.probs2) ? testCase.probs2.map(Number) : [];

  if (expected3.length !== model.numClasses || expected2.length !== model.numBinary) {
    failures += 1;
    details.push(
      `  ✘ #${i} ${JSON.stringify(input)} 期望 probs3 长度 ${model.numClasses} / probs2 长度 ${model.numBinary}，` +
        `实际 ${expected3.length} / ${expected2.length}`,
    );
    continue;
  }

  let got;
  try {
    got = predict(model, input);
  } catch (err) {
    failures += 1;
    details.push(`  ✘ #${i} ${JSON.stringify(input)} 前端推理抛错：${err && err.message ? err.message : err}`);
    continue;
  }

  const diff3 = maxDiff(got.probs3, expected3);
  const diff2 = maxDiff(got.probs2, expected2);
  if (diff3 > maxDiff3) {
    maxDiff3 = diff3;
    worst3 = { input, got: got.probs3, expected: expected3 };
  }
  if (diff2 > maxDiff2) {
    maxDiff2 = diff2;
    worst2 = { input, got: got.probs2, expected: expected2 };
  }
  if (!(diff3 < TOLERANCE) || !(diff2 < TOLERANCE)) {
    failures += 1;
    details.push(
      `  ✘ #${i} ${JSON.stringify(input)} max|Δprobs3|=${diff3.toExponential(3)} max|Δprobs2|=${diff2.toExponential(3)}\n` +
        `     js3=[${got.probs3.map((v) => v.toFixed(6)).join(', ')}]  py3=[${expected3.map((v) => v.toFixed(6)).join(', ')}]\n` +
        `     js2=[${got.probs2.map((v) => v.toFixed(6)).join(', ')}]  py2=[${expected2.map((v) => v.toFixed(6)).join(', ')}]`,
    );
  }
}

console.log('');
console.log(`  用例数：${vectors.cases.length}`);
console.log(`  max|Δprobs3|：${maxDiff3.toExponential(3)}（阈值 ${TOLERANCE}）`);
console.log(`  max|Δprobs2|：${maxDiff2.toExponential(3)}（阈值 ${TOLERANCE}）`);
if (worst3) {
  console.log(
    `  最差 probs3：${JSON.stringify(worst3.input)}  js=[${worst3.got.map((v) => v.toFixed(6)).join(', ')}]  ` +
      `py=[${worst3.expected.map((v) => v.toFixed(6)).join(', ')}]`,
  );
}
if (worst2) {
  console.log(
    `  最差 probs2：${JSON.stringify(worst2.input)}  js=[${worst2.got.map((v) => v.toFixed(6)).join(', ')}]  ` +
      `py=[${worst2.expected.map((v) => v.toFixed(6)).join(', ')}]`,
  );
}
if (details.length) {
  console.log('');
  console.log('  失败明细：');
  console.log(details.slice(0, 30).join('\n'));
  if (details.length > 30) console.log(`  …… 其余 ${details.length - 30} 条省略`);
}

if (failures === 0) {
  console.log(
    `✔ PASS：${vectors.cases.length} 条全部满足 max|Δ| < ${TOLERANCE}（classes=${classes.join('/')}，` +
      `binary=${binaryLabels.join('/')}，格式 ${model.format}）`,
  );
  process.exit(0);
}
console.log(`✘ FAIL：${failures}/${vectors.cases.length} 条超出容差`);
process.exit(1);

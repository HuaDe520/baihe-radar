#!/usr/bin/env node
/**
 * 对拍脚本：JS 前端推理 vs Python 参考实现。
 *
 * 用法（在任意目录执行均可）：
 *   node web/parity_test.mjs
 *       读 web/model.json + train/test_vectors.json，逐条前向，打印最大 |Δprobs|。
 *       test_vectors.json 暂不存在 → 打印「待 lead 生成」并跳过（退出码 0）。
 *   node web/parity_test.mjs --mock
 *       用 web/mock-model.json 代替 web/model.json（本地无真模型时看链路是否通）。
 *   BAIHE_VECTORS=/path/to/vectors.json node web/parity_test.mjs
 *       指定另一份 test vectors（供 CI / 本地临时对拍）。
 *
 * 通过标准（CONTRACT.md 第二节）：max |Δp| < 5e-4，退出码 0；否则退出码 1。
 */

import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildModel, predict } from './inference.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const TOLERANCE = 5e-4;

const useMock = process.argv.includes('--mock');
const modelPath = useMock ? path.join(HERE, 'mock-model.json') : path.join(HERE, 'model.json');
const vectorsPath = process.env.BAIHE_VECTORS
  ? path.resolve(process.env.BAIHE_VECTORS)
  : path.join(ROOT, 'train', 'test_vectors.json');

function readJson(filePath, label) {
  try {
    return JSON.parse(readFileSync(filePath, 'utf8'));
  } catch (err) {
    console.error(`✘ ${label} 解析失败：${filePath}`);
    console.error(`  ${err && err.message ? err.message : err}`);
    process.exit(1);
  }
}

console.log('百合雷达 · JS ↔ Python 推理对拍');
console.log(`  模型：${modelPath}`);
console.log(`  向量：${vectorsPath}`);

if (!existsSync(modelPath)) {
  console.log(`⏳ 未找到模型文件：${modelPath}`);
  console.log('   web/model.json 待 lead 训练导出；本地可先 `node web/make_mock_model.mjs` 生成 mock 权重，');
  console.log('   再用 `node web/parity_test.mjs --mock` 自测链路。跳过对拍，退出码 0。');
  process.exit(0);
}

if (!existsSync(vectorsPath)) {
  console.log('⏳ train/test_vectors.json 待 lead 生成，跳过对拍（退出码 0）。');
  console.log('   生成后重跑本脚本即可：node web/parity_test.mjs');
  process.exit(0);
}

const modelJson = readJson(modelPath, '模型 JSON');
const vectors = readJson(vectorsPath, 'test_vectors.json');

if (!vectors || !Array.isArray(vectors.cases) || vectors.cases.length === 0) {
  console.log('⏳ test_vectors.json 里没有 cases，跳过对拍（退出码 0）。');
  process.exit(0);
}

const model = buildModel(modelJson);
const classes = model.config.classes;
const details = [];
let maxDiff = 0;
let worst = null;
let failures = 0;

for (let i = 0; i < vectors.cases.length; i += 1) {
  const testCase = vectors.cases[i] || {};
  const input = typeof testCase.input === 'string' ? testCase.input : String(testCase.input ?? '');
  const expected = Array.isArray(testCase.probs) ? testCase.probs.map(Number) : [];

  if (expected.length !== model.numClasses) {
    failures += 1;
    details.push(`  ✘ #${i} ${JSON.stringify(input)} 期望 probs 长度 ${model.numClasses}，实际 ${expected.length}`);
    continue;
  }

  let got;
  try {
    got = predict(model, input).probs;
  } catch (err) {
    failures += 1;
    details.push(`  ✘ #${i} ${JSON.stringify(input)} 前端推理抛错：${err && err.message ? err.message : err}`);
    continue;
  }

  let diff = 0;
  for (let c = 0; c < model.numClasses; c += 1) {
    diff = Math.max(diff, Math.abs(got[c] - expected[c]));
  }
  if (diff > maxDiff) {
    maxDiff = diff;
    worst = { input, got, expected };
  }
  if (!(diff < TOLERANCE)) {
    failures += 1;
    details.push(
      `  ✘ #${i} ${JSON.stringify(input)} max|Δp|=${diff.toExponential(3)}  ` +
        `js=[${got.map((v) => v.toFixed(6)).join(', ')}]  py=[${expected.map((v) => v.toFixed(6)).join(', ')}]`,
    );
  }
}

console.log('');
console.log(`  用例数：${vectors.cases.length}`);
console.log(`  max|Δprobs|：${maxDiff.toExponential(3)}（阈值 ${TOLERANCE}）`);
if (worst) {
  console.log(
    `  最差用例：${JSON.stringify(worst.input)}  js=[${worst.got.map((v) => v.toFixed(6)).join(', ')}]  ` +
      `py=[${worst.expected.map((v) => v.toFixed(6)).join(', ')}]`,
  );
}
if (details.length) {
  console.log('');
  console.log('  失败明细：');
  console.log(details.slice(0, 30).join('\n'));
  if (details.length > 30) console.log(`  …… 其余 ${details.length - 30} 条省略`);
}

if (failures === 0) {
  console.log(`✔ PASS：${vectors.cases.length} 条全部满足 max|Δprobs| < ${TOLERANCE}（classes=${classes.join('/')}，格式 ${model.format}）`);
  process.exit(0);
}
console.log(`✘ FAIL：${failures}/${vectors.cases.length} 条超出容差`);
process.exit(1);

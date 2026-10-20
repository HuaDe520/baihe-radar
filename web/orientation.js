/**
 * 百合雷达 BAIHE RADAR · v6「取向套娃」纯前端规则组合器
 *
 * 这是一个**玩笑式的规则组合器**：先用现有 v4 五头模型的 probs（性别头 × 作品头）
 * 拼出 L0 四类取向（直男 / 直女 / 男同 / 女同），再乘上一套转移矩阵做递归套娃，
 * 枚举「当前类型 → ta 喜欢什么类型 → ta 喜欢的类型又喜欢什么类型 …」的所有路径。
 *
 * 转移矩阵来源：
 *   - 优先使用 `./orientation_priors.json`（由训练侧从公开作品角色关系里统计出的
 *     4×4 平滑矩阵，含 source / n_pairs 等元信息）；
 *   - 文件不存在 / 解析失败 / 格式不合法时，静默回落内置的手写娱乐矩阵 TOWER_TRANSITION。
 *
 * ⚠️ 无论哪种矩阵，都是娱乐玩笑；不是模型预测，更不是对任何真人取向的判断。
 *
 * 对外接口（纯函数，无 DOM 依赖，Node 可直接单测）：
 *   buildOrientationL0(probs)                    → number[4] | null（顺序 = TOWER_TYPE_KEYS）
 *   enumerateTowerPaths(l0, depth, matrix?)      → [{types, names, text, rawProb}]（4^(depth+1) 条）
 *   buildOrientationTower(probs, depth=2, opts)  → {depth, l0, paths, samplePool, pathCount, transitionMatrix, isDataMatrix} | null
 *   loadOrientationPriors(url)                   → Promise<priors|null>（永不 reject）
 *   validateOrientationPriors(json)              → priors|null
 *   towerPathText(types)                         → "女同 → 喜欢 女同 → 喜欢 直女"
 *   pickRandomTowerPath(tower, random)           → 从 top50 里随机抽一条路径
 *   formatTowerPercent(prob, digits=1)           → "9.6%"
 */

/** 数据学转移矩阵 JSON 的格式标识 */
export const ORIENTATION_PRIORS_FORMAT = 'orientation-priors-v1';

/** 四类取向的固定顺序：直男 / 直女 / 男同 / 女同 */
export const TOWER_TYPE_KEYS = ['straight_male', 'straight_female', 'gay_male', 'lesbian'];
export const TOWER_TYPE_NAMES = {
  straight_male: '直男',
  straight_female: '直女',
  gay_male: '男同',
  lesbian: '女同',
};

/**
 * 手写转移先验（捏造/娱乐）：行 = 当前类型，列 = ta 喜欢的类型。
 * 列顺序与 TOWER_TYPE_KEYS 一致：直男 / 直女 / 男同 / 女同。
 * 每行和为 1。
 */
export const TOWER_TRANSITION = [
  // 直男: 直女0.60 女同0.15 男同0.10 直男0.15
  [0.15, 0.6, 0.1, 0.15],
  // 直女: 直男0.60 男同0.15 女同0.10 直女0.15
  [0.6, 0.15, 0.15, 0.1],
  // 男同: 男同0.40 直男0.30 直女0.15 女同0.15
  [0.3, 0.15, 0.4, 0.15],
  // 女同: 女同0.40 直女0.30 直男0.15 男同0.15
  [0.15, 0.3, 0.15, 0.4],
];

export const TOWER_DEPTHS = [1, 2, 3];
export const TOWER_TOP_K = 5;
export const TOWER_SAMPLE_POOL = 50;

/** 概率清洗：非有限数/负数一律按 0 处理。 */
function sanitize(value) {
  return Number.isFinite(value) && value > 0 ? value : 0;
}

/** 归一化；总和为 0 时给均匀分布；保证结果有限且和为 1。 */
function normalize(values) {
  const total = values.reduce((sum, value) => sum + value, 0);
  if (!(total > 0)) return values.map(() => 1 / values.length);
  return values.map((value) => value / total);
}

function clampDepth(depth) {
  const n = Math.round(Number(depth));
  if (!Number.isFinite(n)) return 2;
  return Math.min(TOWER_DEPTHS[TOWER_DEPTHS.length - 1], Math.max(TOWER_DEPTHS[0], n));
}

/**
 * 校验并归一化一个 4×4 转移矩阵：每行 4 个非负有限数、行和 > 0，
 * 返回「行归一化」后的新矩阵；不合法返回 null。
 */
export function normalizeTransitionMatrix(matrix) {
  if (!Array.isArray(matrix) || matrix.length !== TOWER_TYPE_KEYS.length) return null;
  const out = [];
  for (const row of matrix) {
    if (!Array.isArray(row) || row.length !== TOWER_TYPE_KEYS.length) return null;
    const values = row.map((value) => sanitize(value));
    const total = values.reduce((sum, value) => sum + value, 0);
    if (!(total > 0)) return null;
    out.push(values.map((value) => value / total));
  }
  return out;
}

/**
 * 校验 `orientation_priors.json` 内容。
 * 至少要求：format = orientation-priors-v1，且 matrix 是合法 4×4 概率矩阵。
 * 额外元信息（source / n_pairs / counts / alpha / generated_at）能做展示就保留。
 *
 * @returns {null | {format, matrix, source, nPairs, alpha, counts, generatedAt}}
 */
export function validateOrientationPriors(json) {
  if (!json || typeof json !== 'object') return null;
  if (json.format !== ORIENTATION_PRIORS_FORMAT) return null;
  const matrix = normalizeTransitionMatrix(json.matrix);
  if (!matrix) return null;
  return {
    format: json.format,
    matrix,
    source: typeof json.source === 'string' ? json.source : '',
    nPairs: Number.isFinite(json.n_pairs) ? json.n_pairs : null,
    alpha: Number.isFinite(json.alpha) ? json.alpha : null,
    counts: Array.isArray(json.counts) ? json.counts : null,
    generatedAt: typeof json.generated_at === 'string' ? json.generated_at : '',
  };
}

/**
 * 读取并校验数据学转移矩阵；**任何失败都静默返回 null**，由调用方回落到手写矩阵。
 * @param {string} url orientation_priors.json 的地址
 * @param {typeof fetch} [fetchImpl] 可注入 fetch（Node 单测用）
 */
export async function loadOrientationPriors(url = './orientation_priors.json', fetchImpl = null) {
  const doFetch = fetchImpl || (typeof fetch === 'function' ? fetch : null);
  if (!doFetch) return null;
  try {
    const response = await doFetch(url, { cache: 'no-store' });
    if (!response || !response.ok) return null;
    let json;
    try {
      json = await response.json();
    } catch (err) {
      return null;
    }
    return validateOrientationPriors(json);
  } catch (err) {
    return null;
  }
}

/**
 * 由五头模型的 probs 推导 L0 四类取向概率（顺序 = TOWER_TYPE_KEYS）：
 *   女同 = pF * pYuri / (pYuri + pBG)
 *   直女 = pF * pBG   / (pYuri + pBG)
 *   男同 = pM * pBL   / (pBL + pBG)
 *   直男 = pM * pBG   / (pBL + pBG)
 * 分母为 0 时该组给 50/50；四项再归一化（总和为 0 时给均匀分布）。
 *
 * @param {{gender?: number[], genre?: number[]}} probs
 * @returns {number[] | null} 长度 4；probs 缺失/不合法时返回 null
 */
export function buildOrientationL0(probs) {
  if (!probs || !Array.isArray(probs.gender) || probs.gender.length < 2) return null;
  if (!Array.isArray(probs.genre) || probs.genre.length < 3) return null;

  const pF = sanitize(probs.gender[0]);
  const pM = sanitize(probs.gender[1]);
  const pYuri = sanitize(probs.genre[0]);
  const pBL = sanitize(probs.genre[1]);
  const pBG = sanitize(probs.genre[2]);

  const femaleDen = pYuri + pBG;
  const maleDen = pBL + pBG;

  const lesbian = femaleDen > 0 ? (pF * pYuri) / femaleDen : pF * 0.5;
  const straightFemale = femaleDen > 0 ? (pF * pBG) / femaleDen : pF * 0.5;
  const gayMale = maleDen > 0 ? (pM * pBL) / maleDen : pM * 0.5;
  const straightMale = maleDen > 0 ? (pM * pBG) / maleDen : pM * 0.5;

  return normalize([straightMale, straightFemale, gayMale, lesbian]);
}

/**
 * 枚举所有套娃路径：路径 = [L0 起点, 第 1 跳目标, ..., 第 depth 跳目标]，
 * 概率 = L0[起点] × ∏ 转移概率，共 4^(depth+1) 条。
 *
 * @param {number[]} l0
 * @param {number} depth
 * @param {number[][]} [matrix=TOWER_TRANSITION] 转移矩阵（行=当前类型，列=喜欢的类型）；
 *        传入非法矩阵时回落到内置手写矩阵
 */
export function enumerateTowerPaths(l0, depth, matrix = TOWER_TRANSITION) {
  if (!Array.isArray(l0) || l0.length !== TOWER_TYPE_KEYS.length) return [];
  const transition = normalizeTransitionMatrix(matrix) || TOWER_TRANSITION;
  const d = clampDepth(depth);
  const paths = [];
  const walk = (types, prob) => {
    if (types.length === d + 1) {
      paths.push({ types: types.slice(), names: types.map((t) => TOWER_TYPE_NAMES[TOWER_TYPE_KEYS[t]]), text: towerPathText(types), rawProb: prob });
      return;
    }
    const last = types[types.length - 1];
    for (let next = 0; next < TOWER_TYPE_KEYS.length; next += 1) {
      walk([...types, next], prob * transition[last][next]);
    }
  };
  for (let start = 0; start < TOWER_TYPE_KEYS.length; start += 1) {
    walk([start], l0[start]);
  }
  return paths;
}

/** 路径 → 人类可读文案：首项平铺，后续加「喜欢」。 */
export function towerPathText(types) {
  if (!Array.isArray(types) || types.length === 0) return '';
  return types
    .map((t, index) => {
      const name = TOWER_TYPE_NAMES[TOWER_TYPE_KEYS[t]] || String(t);
      return index === 0 ? name : `喜欢 ${name}`;
    })
    .join(' → ');
}

/**
 * 构建套娃卡的全部展示数据。
 *
 * @param {object} probs  五头模型 predict() 返回的 probs（需要 gender / genre）
 * @param {number} [depth=2]  套娃深度 1/2/3
 * @param {{matrix?: number[][]}} [options]  可选：数据学转移矩阵（来自 orientation_priors.json）
 * @returns {null | {
 *   depth: number,
 *   l0: Array<{key:string, name:string, prob:number, percent:number}>,
 *   l0Probabilities: number[],
 *   paths: Array<{types:number[], names:string[], text:string, rawProb:number, percent:number}>,
 *   samplePool: Array<{types:number[], names:string[], text:string, rawProb:number}>,
 *   pathCount: number,
 *   topK: number,
 *   transitionMatrix: number[][],
 *   isDataMatrix: boolean,
 * }}
 */
export function buildOrientationTower(probs, depth = 2, options = {}) {
  const l0 = buildOrientationL0(probs);
  if (!l0) return null;
  const d = clampDepth(depth);
  const dataMatrix = options && options.matrix ? normalizeTransitionMatrix(options.matrix) : null;
  const transitionMatrix = dataMatrix || TOWER_TRANSITION;
  const all = enumerateTowerPaths(l0, d, transitionMatrix);

  // 排序：原始连乘概率降序；概率相同按类型序列稳定排序
  const sorted = all.slice().sort((a, b) => {
    if (b.rawProb !== a.rawProb) return b.rawProb - a.rawProb;
    for (let i = 0; i < a.types.length; i += 1) {
      if (a.types[i] !== b.types[i]) return a.types[i] - b.types[i];
    }
    return 0;
  });

  const top = sorted.slice(0, TOWER_TOP_K);
  const topRawSum = top.reduce((sum, path) => sum + path.rawProb, 0);
  const paths = top.map((path) => ({
    ...path,
    percent: topRawSum > 0 ? (path.rawProb / topRawSum) * 100 : 100 / top.length,
  }));

  return {
    depth: d,
    l0: l0.map((prob, index) => ({
      key: TOWER_TYPE_KEYS[index],
      name: TOWER_TYPE_NAMES[TOWER_TYPE_KEYS[index]],
      prob,
      percent: prob * 100,
    })),
    l0Probabilities: l0,
    paths,
    samplePool: sorted.slice(0, Math.min(TOWER_SAMPLE_POOL, sorted.length)),
    pathCount: all.length,
    topK: TOWER_TOP_K,
    transitionMatrix,
    isDataMatrix: Boolean(dataMatrix),
  };
}

/**
 * 从 top50 里随机抽一条路径（「换一个」按钮用）。
 * @param {object} tower buildOrientationTower 结果
 * @param {() => number} [random=Math.random] 可注入随机源便于测试
 */
export function pickRandomTowerPath(tower, random = Math.random) {
  if (!tower || !Array.isArray(tower.samplePool) || tower.samplePool.length === 0) return null;
  const r = typeof random === 'function' ? Number(random()) : Math.random();
  const clamped = Number.isFinite(r) ? Math.min(0.999999, Math.max(0, r)) : 0;
  return tower.samplePool[Math.floor(clamped * tower.samplePool.length)] || null;
}

/** 概率 → "9.6%"；非法值给 "--"。 */
export function formatTowerPercent(prob, digits = 1) {
  if (!Number.isFinite(prob)) return '--';
  return `${(prob * 100).toFixed(digits)}%`;
}

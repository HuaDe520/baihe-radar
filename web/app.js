/**
 * 百合雷达 BAIHE RADAR · 页面逻辑（v4 五头模型）
 *
 * 职责：加载 ./model.json → 绑定输入/示例 chips → 调 inference.js 推理
 *      → 渲染「百合指数」仪表（yuri 头）、性别倾向 / 作品倾向 / ACG 属性 / ACG 百合角色四组概率条、
 *        可切换目标的逐字遮挡贡献、提示与五头模型卡。
 * 纯逻辑部分（verdictFor / modelCardData / buildViewModel …）导出成纯函数，方便 Node 自测。
 */

import { loadModel, predict, occlusionAttribution } from './inference.js';
import {
  TOWER_TYPE_KEYS,
  TOWER_TYPE_NAMES,
  buildOrientationTower,
  loadOrientationPriors,
  pickRandomTowerPath,
  formatTowerPercent,
} from './orientation.js';

export const EXAMPLE_NAMES = [
  '林晚', '苏清', '沈月', '顾言', '张伟', '王芳', '李娜', '陈星',
  'Alice', 'Bob', '林晚_233', 'Alice_777',
  'フォーニ', '渡良瀬 準', 'Watarase Jun',
];
export const DEFAULT_BASELINE = 1 / 3;
/** 二分类头判定阈值：P(百合) ≥ 0.5 视为「判定为百合」。 */
export const BINARY_DECISION_THRESHOLD = 0.5;
export const MODEL_URL = new URL('./model.json', import.meta.url).href;
/** 数据学转移矩阵地址（由训练侧生成；缺失时静默回落到内置娱乐矩阵） */
export const ORIENTATION_PRIORS_URL = new URL('./orientation_priors.json', import.meta.url).href;

/** 五个头的展示顺序与标题（与 config.heads.* 对应） */
export const HEAD_KEYS = ['gender', 'expression', 'genre', 'yuri', 'yuri_char'];
export const HEAD_TITLES = {
  gender: '性别倾向',
  expression: 'ACG 属性',
  genre: '角色取向（虚构作品频道）',
  yuri: '百合判定',
  yuri_char: '百合角色（ACG）',
};
/** 页面概率条展示的四个头（yuri 用大圆环，不重复画条） */
export const BAR_HEAD_KEYS = ['gender', 'genre', 'expression', 'yuri_char'];
/** 逐字贡献可切换的目标头 */
export const CONTRIB_HEAD_KEYS = ['yuri', 'gender', 'expression', 'yuri_char'];

/* ============================================================
   纯函数区（无 DOM 依赖，Node 里可直接测）
   ============================================================ */

/** 12345 → "12,345"（不依赖 Intl，保证各环境一致） */
export function formatInteger(value) {
  if (!Number.isFinite(value)) return '待训练';
  const rounded = Math.round(value);
  const sign = rounded < 0 ? '-' : '';
  const digits = String(Math.abs(rounded));
  return sign + digits.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

/** 0.4123 → "41.2%" */
export function formatPercent(ratio, digits = 1) {
  if (!Number.isFinite(ratio)) return '--';
  return `${(ratio * 100).toFixed(digits)}%`;
}

/** 随机基线：0.333 → "33%" */
export function formatBaseline(ratio) {
  if (!Number.isFinite(ratio) || ratio <= 0) return '33%';
  return `${Math.round(ratio * 100)}%`;
}

/** 遮挡贡献 Δ → "+12.3%" / "-8.0%" / "±0.0%"（单位：百分点） */
export function formatDelta(delta) {
  if (!Number.isFinite(delta)) return '±0.0%';
  const points = delta * 100;
  if (Math.abs(points) < 0.05) return '±0.0%';
  const sign = points > 0 ? '+' : '-';
  return `${sign}${Math.abs(points).toFixed(1)}%`;
}

/**
 * 按「百合指数」（yuri 头 P(百合)）分档的一句话点评，共 8 档，好玩不低俗。
 * 沿用 v2 文案并微调措辞。
 */
export function verdictFor(baiheIndex) {
  if (!Number.isFinite(baiheIndex)) return '雷达正在重启，稍后再扫一次。';
  if (baiheIndex >= 0.9) return '雷达爆表——这个名字在晋江百合频道是有编制的。';
  if (baiheIndex >= 0.8) return '警报拉满，建议立刻给这个名字安排一位女主角。';
  if (baiheIndex >= 0.7) return '粉紫信号明显，空气里已经开始飘百合花瓣了。';
  if (baiheIndex >= 0.6) return '雷达持续嗡鸣，粉紫色成分肉眼可见。';
  if (baiheIndex >= 0.45) return '雷达左摇右摆：薛定谔的百合，全看作者怎么落笔。';
  if (baiheIndex >= 0.3) return '雷达有点犹豫：友情以上、恋爱未满。';
  if (baiheIndex >= 0.15) return '雷达偏安静，这个名字更适合友情出演。';
  return '雷达安静得像深夜的图书馆。';
}

/** 仪表分档：返回 {label, tier}，tier 用于 CSS 配色（hot / warm / cool）。 */
export function gaugeTier(baiheIndex) {
  if (!Number.isFinite(baiheIndex)) return { label: '雷达待机', tier: 'idle' };
  if (baiheIndex >= 0.9) return { label: '雷达爆表', tier: 'hot' };
  if (baiheIndex >= 0.8) return { label: '警报拉满', tier: 'hot' };
  if (baiheIndex >= 0.7) return { label: '信号强烈', tier: 'warm' };
  if (baiheIndex >= 0.6) return { label: '持续嗡鸣', tier: 'warm' };
  if (baiheIndex >= 0.45) return { label: '摇摆不定', tier: 'warm' };
  if (baiheIndex >= 0.3) return { label: '有点犹豫', tier: 'cool' };
  if (baiheIndex >= 0.15) return { label: '雷达偏静', tier: 'cool' };
  return { label: '雷达静默', tier: 'cool' };
}

/** 某个头的概率 → 展示用 items（类名取 config.names_zh）。 */
export function headItems(model, probs, headKey) {
  const head = model.heads[headKey];
  const values = probs[headKey];
  let topIndex = 0;
  for (let i = 1; i < values.length; i += 1) if (values[i] > values[topIndex]) topIndex = i;
  return {
    key: headKey,
    title: HEAD_TITLES[headKey] || headKey,
    items: values.map((prob, index) => ({
      index,
      classKey: head.classes[index],
      name: head.namesZh[index] || head.classes[index],
      prob,
      percent: prob * 100,
      isTop: index === topIndex,
    })),
    topIndex,
    top: {
      index: topIndex,
      name: head.namesZh[topIndex] || head.classes[topIndex],
      percent: values[topIndex] * 100,
    },
  };
}

/** 逐字贡献目标的展示名。 */
export function contributionLabel(headKey, className) {
  if (headKey === 'yuri') return '百合指数';
  if (headKey === 'gender') return `性别「${className}」`;
  if (headKey === 'expression') return `ACG 属性「${className}」`;
  if (headKey === 'genre') return `作品倾向「${className}」`;
  if (headKey === 'yuri_char') return `百合角色「${className}」`;
  return `${headKey}「${className}」`;
}

/**
 * 从模型对象里整理模型卡要用的数字。
 * 指标缺失（或 mock 权重 n = 0）时返回 null，页面显示「待训练」。
 */
export function modelCardData(model) {
  const metrics = model && model.metrics ? model.metrics : null;
  const pick = (headKey, field) => {
    const head = metrics && metrics[headKey];
    if (!head || !Number.isFinite(head.n) || head.n <= 0) return null;
    const value = head[field];
    return Number.isFinite(value) && value >= 0 ? value : null;
  };
  const headN = (headKey) => {
    const head = metrics && metrics[headKey];
    return head && Number.isFinite(head.n) && head.n > 0 ? head.n : null;
  };
  const hasTraining = HEAD_KEYS.some((key) => headN(key) != null);
  const genreBaseline =
    metrics && metrics.genre && Number.isFinite(metrics.genre.baseline) && metrics.genre.baseline > 0
      ? metrics.genre.baseline
      : DEFAULT_BASELINE;

  const dataset = model && model.dataset ? model.dataset : null;
  const datasetCounts = {
    genderZh: dataset && Number.isFinite(dataset.gender_zh) ? dataset.gender_zh : 0,
    genderForeign: dataset && Number.isFinite(dataset.gender_foreign) ? dataset.gender_foreign : 0,
    expression: dataset && Number.isFinite(dataset.expression) ? dataset.expression : 0,
    genre: dataset && Number.isFinite(dataset.genre) ? dataset.genre : 0,
    yuriChar: dataset && Number.isFinite(dataset.yuri_char) ? dataset.yuri_char : 0,
  };
  const hasDataset = Object.values(datasetCounts).some((value) => value > 0);

  return {
    hasTraining,
    gender: {
      acc: pick('gender', 'acc'),
      balancedAcc: pick('gender', 'balanced_acc'),
      n: headN('gender'),
    },
    expression: {
      acc: pick('expression', 'acc'),
      balancedAcc: pick('expression', 'balanced_acc'),
      n: headN('expression'),
    },
    genre: {
      acc: pick('genre', 'acc'),
      baseline: genreBaseline,
      n: headN('genre'),
    },
    yuri: {
      balancedAcc: pick('yuri', 'balanced_acc'),
      n: headN('yuri'),
    },
    yuriChar: {
      acc: pick('yuri_char', 'acc'),
      balancedAcc: pick('yuri_char', 'balanced_acc'),
      n: headN('yuri_char'),
    },
    dataset: datasetCounts,
    hasDataset,
    format: model && typeof model.format === 'string' ? model.format : '',
  };
}

/**
 * 一次扫描的完整视图数据：五头概率、点评、仪表档位、各组概率条、逐字贡献、提示。
 * @param {object} model       buildModel / loadModel 结果
 * @param {string} rawInput    原始输入
 * @param {string} [contribHead='yuri']        逐字贡献的目标头
 * @param {number|null} [contribClassIndex]    逐字贡献的目标类别（缺省自动）
 */
export function buildViewModel(model, rawInput, contribHead = 'yuri', contribClassIndex = null) {
  const result = predict(model, rawInput);
  const attribution = occlusionAttribution(model, rawInput, contribHead, contribClassIndex);
  const probs = result.probs;
  const baiheIndex = result.baiheIndex;
  const genreHead = headItems(model, probs, 'genre');

  const target = attribution.length
    ? {
        head: attribution[0].head,
        classIndex: attribution[0].classIndex,
        className: attribution[0].className,
        label: contributionLabel(attribution[0].head, attribution[0].className),
        baseProb: probs[attribution[0].head][attribution[0].classIndex],
      }
    : { head: contribHead, classIndex: 0, className: '', label: contributionLabel(contribHead, ''), baseProb: 0 };

  const notes = [];
  if (result.truncated) {
    notes.push({
      icon: '✂️',
      text: `名字有 ${result.inputLength} 个字符，超过 ${model.config.max_len} 个字符的上限，只取前 ${model.config.max_len} 个字符参与扫描。`,
    });
  }
  if (result.unknownChars.length > 0) {
    const unique = [...new Set(result.unknownChars)];
    notes.push({
      icon: '⚠️',
      text: `有 ${result.unknownChars.length} 个字不在模型字表里（${unique.join('、')}），已按「未知字」处理，结果可能不太准。`,
    });
  }

  return {
    input: String(rawInput ?? ''),
    chars: result.chars,
    ids: result.ids,
    truncated: result.truncated,
    inputLength: result.inputLength,
    unknownChars: result.unknownChars,
    maxLen: model.config.max_len,
    probs,
    baiheIndex,
    baihePercent: baiheIndex * 100,
    baiheRounded: Math.round(baiheIndex * 100),
    binaryDecision: baiheIndex >= BINARY_DECISION_THRESHOLD ? 'baihe' : 'not_baihe',
    verdict: verdictFor(baiheIndex),
    gauge: gaugeTier(baiheIndex),
    headData: {
      gender: headItems(model, probs, 'gender'),
      expression: headItems(model, probs, 'expression'),
      genre: genreHead,
      yuri_char: headItems(model, probs, 'yuri_char'),
    },
    genreLeader: genreHead.top,
    contribution: { ...target, items: attribution },
    notes,
  };
}

/* ============================================================
   DOM 区（只在浏览器里执行；Node import 时直接跳过）
   ============================================================ */

function initApp() {
  const $ = (selector) => document.querySelector(selector);

  const els = {
    form: $('#scan-form'),
    input: $('#name-input'),
    button: $('#scan-btn'),
    buttonText: $('#scan-btn-text'),
    hint: $('#input-hint'),
    chips: $('#example-chips'),
    loading: $('#loading-card'),
    error: $('#error-card'),
    errorTitle: $('#error-title'),
    errorText: $('#error-text'),
    retry: $('#retry-load'),
    result: $('#result-card'),
    gauge: $('#gauge'),
    gaugeMeter: $('#gauge-meter'),
    gaugeNumber: $('#gauge-number'),
    gaugeTier: $('#gauge-tier'),
    verdict: $('#verdict-text'),
    leader: $('#leader-text'),
    notes: $('#scan-notes'),
    dims: $('#dims'),
    bars: {},
    barContainers: {
      gender: $('#gender-bars'),
      expression: $('#expression-bars'),
      genre: $('#genre-bars'),
      yuri_char: $('#yuri-char-bars'),
    },
    contribCard: $('#contrib-card'),
    contribList: $('#contrib-list'),
    contribTarget: $('#contrib-target'),
    segButtons: Array.from(document.querySelectorAll('[data-contrib-head]')),
    towerCard: $('#tower-card'),
    towerL0Bars: $('#tower-l0-bars'),
    towerList: $('#tower-list'),
    towerPick: $('#tower-pick'),
    towerShuffle: $('#tower-shuffle'),
    towerDepthButtons: Array.from(document.querySelectorAll('[data-tower-depth]')),
    towerNote: $('#tower-note'),
    towerDisclaimer: $('#tower-disclaimer'),
    towerSource: $('#tower-source'),
    metricIds: [
      'metric-gender-acc',
      'metric-gender-bal',
      'metric-gender-n',
      'metric-expression-acc',
      'metric-expression-bal',
      'metric-expression-n',
      'metric-genre-acc',
      'metric-genre-base',
      'metric-genre-n',
      'metric-yuri-bal',
      'metric-yuri-n',
      'metric-yuri-char-acc',
      'metric-yuri-char-bal',
      'metric-yuri-char-n',
    ],
    metricEls: {},
    datasetLine: $('#dataset-line'),
    formatTag: $('#format-tag'),
  };
  els.metricIds.forEach((id) => {
    els.metricEls[id] = $(`#${id}`);
  });

  if (!els.form || !els.input || !els.result) return;

  let model = null;
  let busy = false;
  let composing = false;
  let lastRaw = '';
  let lastView = null;
  let contribHead = 'yuri';
  let towerDepth = 2;
  /** 数据学转移矩阵（orientation_priors.json）；null = 用内置手写娱乐矩阵 */
  let orientationPriors = null;
  /** @type {Record<string, Array<{row: HTMLElement, pct: HTMLElement, fill: HTMLElement}>>} */
  const barRefs = {};
  /** L0 四条小概率条的 DOM 引用（顺序 = TOWER_TYPE_KEYS） */
  const towerBarRefs = [];

  /* ---------- 初始骨架 ---------- */
  initGauge();
  renderChips();
  buildTowerBars();
  applyTowerCopy();
  bindEvents();
  bootModel();
  loadOrientationPriors(ORIENTATION_PRIORS_URL).then((priors) => {
    orientationPriors = priors;
    applyTowerCopy();
    if (lastView) renderTower(lastView);
  });

  function initGauge() {
    if (!els.gaugeMeter) return;
    const radius = els.gaugeMeter.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;
    els.gaugeMeter.style.strokeDasharray = `${circumference} ${circumference}`;
    els.gaugeMeter.style.strokeDashoffset = `${circumference}`;
  }

  function renderChips() {
    if (!els.chips) return;
    let buttons = Array.from(els.chips.querySelectorAll('[data-name]'));
    // 兜底：HTML 里没有静态 chips 时用 JS 补出来
    if (buttons.length === 0) {
      buttons = EXAMPLE_NAMES.map((name) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'chip';
        button.dataset.name = name;
        button.textContent = name;
        els.chips.append(button);
        return button;
      });
    }
    buttons.forEach((button) => {
      const name = button.dataset.name || button.textContent || '';
      button.setAttribute('aria-label', `试试「${name}」`);
      button.addEventListener('click', () => {
        els.input.value = name;
        if (model) {
          handleScan();
        } else {
          setHint('模型还在加载，稍等一下下…');
        }
      });
    });
  }

  function bindEvents() {
    els.form.addEventListener('submit', (event) => {
      event.preventDefault();
      if (composing) return;
      handleScan();
    });

    els.input.addEventListener('input', () => {
      hideHint();
    });

    els.input.addEventListener('compositionstart', () => {
      composing = true;
    });
    els.input.addEventListener('compositionend', () => {
      composing = false;
    });

    if (els.retry) {
      els.retry.addEventListener('click', () => {
        bootModel();
      });
    }

    els.segButtons.forEach((button) => {
      button.addEventListener('click', () => {
        const head = button.dataset.contribHead;
        if (!CONTRIB_HEAD_KEYS.includes(head) || !model || !lastRaw) return;
        setContribHead(head);
        try {
          const view = buildViewModel(model, lastRaw, head);
          renderContribution(view);
        } catch (err) {
          setHint(err && err.message ? err.message : '切换贡献视角失败，请再扫一次。');
        }
      });
    });

    // 套娃深度切换 / 随机抽一条
    els.towerDepthButtons.forEach((button) => {
      button.addEventListener('click', () => {
        setTowerDepth(Number(button.dataset.towerDepth));
      });
    });
    if (els.towerShuffle) {
      els.towerShuffle.addEventListener('click', () => {
        handleTowerShuffle();
      });
    }
  }

  function setContribHead(head) {
    contribHead = head;
    els.segButtons.forEach((button) => {
      const active = button.dataset.contribHead === head;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
  }

  /* ---------- 模型加载 ---------- */
  async function bootModel() {
    hideError();
    setLoading(true);
    setScanEnabled(false, '加载模型中…');
    try {
      model = await loadModel(MODEL_URL);
      buildHeadBars();
      renderModelCard();
      setLoading(false);
      setScanEnabled(true, '开始扫描');
      if (els.input) els.input.placeholder = '输入中文 / 英文 / 日文名字或网名，如：林晚、Alice_777、フォーニ';
    } catch (err) {
      model = null;
      setLoading(false);
      setScanEnabled(false, '开始扫描');
      els.result.hidden = true;
      if (els.dims) els.dims.hidden = true;
      if (els.contribCard) els.contribCard.hidden = true;
      if (els.towerCard) els.towerCard.hidden = true;
      showError('模型加载失败', err && err.message ? err.message : String(err));
    }
  }

  /** 按模型 config 动态生成三组概率条（类名/中文名都来自模型）。 */
  function buildHeadBars() {
    if (!model) return;
    BAR_HEAD_KEYS.forEach((headKey) => {
      const container = els.barContainers[headKey];
      if (!container) return;
      const head = model.heads[headKey];
      container.replaceChildren();
      barRefs[headKey] = head.classes.map((classKey, index) => {
        const cssKey = String(classKey).replace(/[^a-zA-Z0-9_-]/g, '-');
        const row = document.createElement('div');
        row.className = `bar bar--${headKey}-${cssKey}`;

        const headEl = document.createElement('div');
        headEl.className = 'bar__head';
        const nameEl = document.createElement('span');
        nameEl.className = 'bar__name';
        const dot = document.createElement('span');
        dot.className = `bar__dot bar__dot--${headKey}-${cssKey}`;
        dot.setAttribute('aria-hidden', 'true');
        nameEl.append(dot, document.createTextNode(head.namesZh[index] || classKey));
        const pct = document.createElement('span');
        pct.className = 'bar__pct';
        pct.textContent = '--%';
        headEl.append(nameEl, pct);

        const track = document.createElement('div');
        track.className = 'bar__track';
        const fill = document.createElement('span');
        fill.className = `bar__fill bar__fill--${headKey}-${cssKey}`;
        track.append(fill);

        row.append(headEl, track);
        container.append(row);
        return { row, pct, fill };
      });
    });
  }

  function renderModelCard() {
    if (!model) return;
    const data = modelCardData(model);
    const set = (id, value) => {
      const el = els.metricEls[id];
      if (el) el.textContent = value;
    };
    const acc = (value) => (value == null ? '待训练' : formatPercent(value, 1));
    const count = (value) => (value == null ? '待训练' : formatInteger(value));

    set('metric-gender-acc', acc(data.gender.acc));
    set('metric-gender-bal', acc(data.gender.balancedAcc));
    set('metric-gender-n', count(data.gender.n));
    set('metric-expression-acc', acc(data.expression.acc));
    set('metric-expression-bal', acc(data.expression.balancedAcc));
    set('metric-expression-n', count(data.expression.n));
    set('metric-genre-acc', acc(data.genre.acc));
    set('metric-genre-base', formatBaseline(data.genre.baseline));
    set('metric-genre-n', count(data.genre.n));
    set('metric-yuri-bal', acc(data.yuri.balancedAcc));
    set('metric-yuri-n', count(data.yuri.n));
    set('metric-yuri-char-acc', acc(data.yuriChar.acc));
    set('metric-yuri-char-bal', acc(data.yuriChar.balancedAcc));
    set('metric-yuri-char-n', count(data.yuriChar.n));

    if (els.formatTag && data.format) els.formatTag.textContent = data.format;
    if (els.datasetLine) {
      if (!data.hasDataset) {
        els.datasetLine.textContent = '训练数据：待接入';
      } else {
        const parts = [
          `中文姓名 ${formatInteger(data.dataset.genderZh)}`,
          `外国姓名 ${formatInteger(data.dataset.genderForeign)}`,
          `ACG 属性 ${formatInteger(data.dataset.expression)}`,
          `小说主角名 ${formatInteger(data.dataset.genre)}`,
        ];
        if (data.dataset.yuriChar > 0) parts.push(`百合角色 ${formatInteger(data.dataset.yuriChar)}`);
        els.datasetLine.textContent = `训练数据：${parts.join(' · ')}`;
      }
    }
  }

  /* ---------- 扫描 ---------- */
  async function handleScan() {
    if (busy) return;

    const raw = els.input.value;
    if (!raw || !raw.replace(/\s+/g, '')) {
      setHint('请先输入一个名字或网名，比如「林晚」「Alice_777」或「フォーニ」');
      shakeInput();
      els.input.focus();
      return;
    }
    if (!model) {
      setHint('模型还没加载好，稍等一下下…');
      return;
    }

    busy = true;
    hideHint();
    setScanning(true);
    try {
      // 留出一帧时间让「扫描中」动效可见，计算本身是毫秒级的
      await new Promise((resolve) => setTimeout(resolve, 320));
      lastRaw = raw;
      setContribHead('yuri');
      const view = buildViewModel(model, raw, 'yuri');
      renderResult(view);
    } catch (err) {
      setHint(err && err.message ? err.message : '扫描时出了点问题，请再试一次。');
    } finally {
      setScanning(false);
      busy = false;
    }
  }

  function renderResult(view) {
    els.result.hidden = false;
    if (els.dims) els.dims.hidden = false;
    if (els.contribCard) els.contribCard.hidden = false;
    els.result.classList.remove('pop');
    // 强制重排，让入场动画每次都能重播
    void els.result.offsetWidth;
    els.result.classList.add('pop');

    // 仪表：百合指数 = yuri 头 P(百合)
    const ratio = Math.min(1, Math.max(0, view.baiheIndex));
    const radius = els.gaugeMeter.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;
    els.gaugeMeter.style.strokeDashoffset = `${circumference * (1 - ratio)}`;
    els.gauge.dataset.tier = view.gauge.tier;
    els.gauge.setAttribute('aria-label', `百合指数 ${view.baiheRounded}%`);
    els.gaugeNumber.textContent = String(view.baiheRounded);
    els.gaugeTier.textContent = view.gauge.label;

    // 点评 + 各头摘要
    els.verdict.textContent = view.verdict;
    const decisionText = view.binaryDecision === 'baihe' ? '判定为「百合」' : '判定为「非百合」';
    els.leader.textContent =
      `百合判定：${decisionText}（P(百合)=${formatPercent(view.baiheIndex, 1)}）` +
      ` · 作品头最像：${view.genreLeader.name}（${view.genreLeader.percent.toFixed(1)}%）`;

    // 性别 / ACG 属性 / 作品 三组概率条
    BAR_HEAD_KEYS.forEach((headKey) => {
      const refs = barRefs[headKey];
      const headData = view.headData[headKey];
      if (!refs || !headData) return;
      headData.items.forEach((item, index) => {
        const ref = refs[index];
        if (!ref) return;
        const percent = Math.min(1, Math.max(0, item.prob));
        ref.pct.textContent = formatPercent(percent, 1);
        ref.fill.style.width = `${percent * 100}%`;
        ref.row.classList.toggle('is-top', item.isTop);
      });
    });

    // 提示（超长截断 / 未登录字）
    els.notes.replaceChildren();
    if (view.notes.length > 0) {
      view.notes.forEach((note) => {
        const li = document.createElement('li');
        const icon = document.createElement('span');
        icon.className = 'note__icon';
        icon.setAttribute('aria-hidden', 'true');
        icon.textContent = note.icon;
        const text = document.createElement('span');
        text.textContent = note.text;
        li.append(icon, text);
        els.notes.append(li);
      });
      els.notes.hidden = false;
    } else {
      els.notes.hidden = true;
    }

    // 逐字贡献
    renderContribution(view);

    // 取向套娃（纯前端规则组合器）
    lastView = view;
    renderTower(view);

    // 移动端：结果在视口外时轻轻滚过去
    if (typeof window !== 'undefined' && els.result.getBoundingClientRect().top > window.innerHeight * 0.72) {
      els.result.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  function renderContribution(view) {
    if (els.contribList) els.contribList.replaceChildren();
    if (els.contribTarget) {
      const target = view.contribution;
      const percent = Number.isFinite(target.baseProb) ? formatPercent(target.baseProb, 1) : '--';
      els.contribTarget.textContent = `当前目标：${target.label}（P=${percent}）`;
    }
    if (!els.contribList) return;
    view.contribution.items.forEach((item) => {
      const points = item.delta * 100;
      const chip = document.createElement('span');
      chip.className = 'contrib-chip';
      if (points > 0.05) chip.classList.add('contrib-chip--pos');
      else if (points < -0.05) chip.classList.add('contrib-chip--neg');
      chip.title = `把「${item.char}」换成未知字后，${view.contribution.label} ${formatDelta(item.delta)}（单位：百分点）`;

      const charEl = document.createElement('span');
      charEl.className = 'contrib-chip__char';
      charEl.textContent = item.char;

      const deltaEl = document.createElement('span');
      deltaEl.className = 'contrib-chip__delta';
      deltaEl.textContent = formatDelta(item.delta);

      chip.append(charEl, deltaEl);
      els.contribList.append(chip);
    });
  }

  /* ---------- 取向套娃（纯前端规则组合器） ---------- */

  /** 生成 L0 四条小概率条（类型固定：直男/直女/男同/女同）。 */
  function buildTowerBars() {
    if (!els.towerL0Bars) return;
    els.towerL0Bars.replaceChildren();
    towerBarRefs.length = 0;
    TOWER_TYPE_KEYS.forEach((key) => {
      const cssKey = key.replace(/_/g, '-');
      const row = document.createElement('div');
      row.className = `bar bar--tower-${cssKey}`;

      const headEl = document.createElement('div');
      headEl.className = 'bar__head';
      const nameEl = document.createElement('span');
      nameEl.className = 'bar__name';
      const dot = document.createElement('span');
      dot.className = `bar__dot bar__dot--tower-${cssKey}`;
      dot.setAttribute('aria-hidden', 'true');
      nameEl.append(dot, document.createTextNode(TOWER_TYPE_NAMES[key]));
      const pct = document.createElement('span');
      pct.className = 'bar__pct';
      pct.textContent = '--%';
      headEl.append(nameEl, pct);

      const track = document.createElement('div');
      track.className = 'bar__track';
      const fill = document.createElement('span');
      fill.className = `bar__fill bar__fill--tower-${cssKey}`;
      track.append(fill);

      row.append(headEl, track);
      els.towerL0Bars.append(row);
      towerBarRefs.push({ row, pct, fill });
    });
  }

  /** 当前应使用的转移矩阵选项：有数据先验用数据矩阵，否则空对象（回落内置娱乐矩阵）。 */
  function towerMatrixOptions() {
    if (orientationPriors && orientationPriors.matrix) return { matrix: orientationPriors.matrix };
    return {};
  }

  /** 根据是否加载到数据先验，切换套娃卡的说明/免责/矩阵来源文案。 */
  function applyTowerCopy() {
    const priors = orientationPriors;
    if (els.towerNote) {
      els.towerNote.textContent = priors
        ? '把「性别用字倾向 × 虚构作品取向」按数据估计的转移矩阵递归套娃（样本来自公开作品角色关系统计），纯属捏造与娱乐，不是真人取向判断。'
        : '把「性别用字倾向 × 虚构作品取向」按内置娱乐矩阵递归套娃，纯属捏造与娱乐，不是真人取向判断。';
    }
    if (els.towerDisclaimer) {
      els.towerDisclaimer.textContent = priors
        ? '套娃链是规则组合器按数据估计的转移矩阵生成的玩笑（受 B 站《更多性取向》启发），不是模型预测，更不是真人取向判断。'
        : '套娃链是规则组合器按手写先验生成的玩笑（受 B 站《更多性取向》启发），不是模型预测，更不是真人取向判断。';
    }
    if (els.towerSource) {
      if (priors && priors.matrix) {
        const n = Number.isFinite(priors.nPairs) ? `${formatInteger(priors.nPairs)} 对样本` : '样本数未知';
        const source = priors.source ? ` · 来源 ${priors.source}` : '';
        els.towerSource.textContent = `转移矩阵：数据估计（${n}${source}）`;
      } else {
        els.towerSource.textContent = '转移矩阵：内置娱乐矩阵（未加载数据先验）';
      }
      els.towerSource.hidden = false;
    }
  }

  /** 渲染套娃卡：L0 概率条 + top5 路径（百分比为 top5 归一化）。 */
  function renderTower(view) {
    if (!els.towerCard) return;
    let tower = null;
    try {
      tower = buildOrientationTower(view.probs, towerDepth, towerMatrixOptions());
    } catch (err) {
      tower = null;
    }
    if (!tower) {
      els.towerCard.hidden = true;
      return;
    }
    els.towerCard.hidden = false;

    // L0 四条小概率条
    let topL0 = 0;
    tower.l0.forEach((item, index) => {
      if (item.prob > tower.l0[topL0].prob) topL0 = index;
    });
    tower.l0.forEach((item, index) => {
      const ref = towerBarRefs[index];
      if (!ref) return;
      const prob = Math.min(1, Math.max(0, item.prob));
      ref.pct.textContent = formatTowerPercent(prob, 1);
      ref.fill.style.width = `${prob * 100}%`;
      ref.row.classList.toggle('is-top', index === topL0);
    });

    // top5 链
    if (els.towerList) {
      els.towerList.replaceChildren();
      tower.paths.forEach((path) => {
        const li = document.createElement('li');
        li.className = 'tower__item';
        li.title = `原始连乘值 ${path.rawProb.toExponential(3)}；展示百分比按 top${tower.paths.length} 归一化`;
        const chain = document.createElement('span');
        chain.className = 'tower__chain';
        chain.textContent = path.text;
        const pct = document.createElement('span');
        pct.className = 'tower__pct';
        pct.textContent = `${path.percent.toFixed(1)}%`;
        li.append(chain, pct);
        els.towerList.append(li);
      });
    }

    // 切深度/换名字后先清空抽签行
    if (els.towerPick) {
      els.towerPick.hidden = true;
      els.towerPick.textContent = '';
    }
    updateTowerDepthButtons();
  }

  function setTowerDepth(depth) {
    const n = Math.round(Number(depth));
    towerDepth = Number.isFinite(n) ? Math.min(3, Math.max(1, n)) : 2;
    updateTowerDepthButtons();
    if (lastView) renderTower(lastView);
  }

  function updateTowerDepthButtons() {
    els.towerDepthButtons.forEach((button) => {
      const active = Number(button.dataset.towerDepth) === towerDepth;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
  }

  /** 「换一个」：从 top50 路径里随机抽一条显示（纯娱乐）。 */
  function handleTowerShuffle() {
    if (!els.towerPick || !lastView) return;
    let tower = null;
    try {
      tower = buildOrientationTower(lastView.probs, towerDepth, towerMatrixOptions());
    } catch (err) {
      tower = null;
    }
    const path = pickRandomTowerPath(tower);
    if (!path) return;
    els.towerPick.hidden = false;
    els.towerPick.textContent = `🎲 随机抽中（top50 内）：${path.text} · 原始概率 ${(path.rawProb * 100).toFixed(3)}%`;
  }

  /* ---------- 状态与提示 ---------- */
  function setScanEnabled(enabled, text) {
    if (els.button) els.button.disabled = !enabled;
    if (els.buttonText) els.buttonText.textContent = text;
  }

  function setScanning(scanning) {
    if (typeof document !== 'undefined') document.body.classList.toggle('is-scanning', scanning);
    if (scanning) setScanEnabled(false, '扫描中…');
    else if (model) setScanEnabled(true, '开始扫描');
  }

  function setLoading(visible) {
    if (els.loading) els.loading.hidden = !visible;
  }

  function setHint(message) {
    if (els.hint) els.hint.textContent = message;
  }

  function hideHint() {
    setHint('');
  }

  function shakeInput() {
    els.input.classList.remove('shake');
    void els.input.offsetWidth;
    els.input.classList.add('shake');
  }

  function showError(title, message) {
    if (!els.error) return;
    els.error.hidden = false;
    if (els.errorTitle) els.errorTitle.textContent = title;
    if (els.errorText) els.errorText.textContent = message;
  }

  function hideError() {
    if (els.error) els.error.hidden = true;
  }
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp, { once: true });
  } else {
    initApp();
  }
}

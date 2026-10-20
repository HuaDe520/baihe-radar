/**
 * 百合雷达 BAIHE RADAR · 页面逻辑（v2 双头模型）
 *
 * 职责：加载 ./model.json → 绑定输入/示例 chips → 调 inference.js 推理
 *      → 渲染「百合指数」仪表（二分类头）、三条三类概率条（三分类头）、
 *        逐字遮挡贡献（基于百合指数）、提示与模型卡。
 * 纯逻辑部分（verdictFor / modelCardData / buildViewModel …）导出成纯函数，方便 Node 自测。
 */

import { loadModel, predict, occlusionAttribution } from './inference.js';

export const EXAMPLE_NAMES = ['林晚', '苏清', '沈月', '顾言', '张伟', '王芳', '李娜', '陈星'];
export const DEFAULT_BASELINE = 1 / 3;
export const CLASS_KEYS = ['baihe', 'purelove', 'romance'];
/** 二分类头判定阈值：P(百合) ≥ 0.5 视为「判定为百合」。 */
export const BINARY_DECISION_THRESHOLD = 0.5;
export const MODEL_URL = new URL('./model.json', import.meta.url).href;

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
 * 按「百合指数」（二分类头 P(百合)）分档的一句话点评，共 8 档，好玩不低俗。
 * 前几档沿用 v1 文案，低分段细分了两档。
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

/**
 * 从模型对象里整理模型卡要用的数字。
 * 指标缺失（或 mock 权重 n_train = 0）时返回 null，页面显示「待训练」。
 */
export function modelCardData(model) {
  const metrics = model && model.metrics ? model.metrics : null;
  const hasTraining = Boolean(metrics && Number.isFinite(metrics.n_train) && metrics.n_train > 0);
  const nTrain = hasTraining ? metrics.n_train : null;
  const testAcc3 =
    hasTraining && Number.isFinite(metrics.test_acc_3cls) && metrics.test_acc_3cls >= 0
      ? metrics.test_acc_3cls
      : null;
  const binaryBalancedAcc =
    hasTraining && Number.isFinite(metrics.binary_balanced_acc) && metrics.binary_balanced_acc >= 0
      ? metrics.binary_balanced_acc
      : null;
  const binaryAcc =
    hasTraining && Number.isFinite(metrics.binary_acc) && metrics.binary_acc >= 0 ? metrics.binary_acc : null;
  const baseline =
    metrics && Number.isFinite(metrics.baseline_3cls) && metrics.baseline_3cls > 0
      ? metrics.baseline_3cls
      : DEFAULT_BASELINE;

  const dataset = model && model.dataset ? model.dataset : null;
  const perLabel = CLASS_KEYS.map((key) => (dataset && Number.isFinite(dataset[key]) ? dataset[key] : 0));
  const hasDataset = perLabel.some((value) => value > 0);

  return {
    nTrain,
    testAcc3,
    binaryBalancedAcc,
    binaryAcc,
    baseline,
    hasTraining,
    dataset,
    perLabel,
    hasDataset,
    format: model && typeof model.format === 'string' ? model.format : '',
  };
}

/**
 * 一次扫描的完整视图数据：双头概率、点评、仪表档位、逐字贡献、提示。
 * @param {object} model     buildModel / loadModel 结果
 * @param {string} rawInput  原始输入
 */
export function buildViewModel(model, rawInput) {
  const result = predict(model, rawInput);
  const attribution = occlusionAttribution(model, rawInput);

  const probs3 = result.probs3.slice();
  const probs2 = result.probs2.slice();
  const classNames = model.config.class_names_zh.slice();
  const binaryLabels = model.config.binary_labels.slice();
  const baiheIndex = result.baiheIndex; // = probs2[百合位] = P(百合)
  const baihe3 = probs3[model.baiheIndex3]; // 三分类头给的 P(百合)
  const top3Index = probs3.indexOf(Math.max(...probs3));

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

  const binaryDecision = baiheIndex >= BINARY_DECISION_THRESHOLD ? 'baihe' : 'not_baihe';

  return {
    input: String(rawInput ?? ''),
    chars: result.chars,
    ids: result.ids,
    truncated: result.truncated,
    inputLength: result.inputLength,
    unknownChars: result.unknownChars,
    maxLen: model.config.max_len,
    probs3,
    probs2,
    classNames,
    binaryLabels,
    baiheIndex3: model.baiheIndex3,
    baihe3,
    baiheIndex,
    baihePercent: baiheIndex * 100,
    baiheRounded: Math.round(baiheIndex * 100),
    binaryDecision,
    verdict: verdictFor(baiheIndex),
    gauge: gaugeTier(baiheIndex),
    leader: {
      name: classNames[top3Index] || '未知',
      percent: probs3[top3Index] * 100,
    },
    attribution,
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
    contrib: $('#contrib-list'),
    bars: {},
    pct: {},
    fills: {},
    statNTrain: $('#stat-n-train'),
    statTestAcc: $('#stat-test-acc'),
    statBinBalanced: $('#stat-bin-balanced-acc'),
    statBaseline: $('#stat-baseline'),
    datasetLine: $('#dataset-line'),
    formatTag: $('#format-tag'),
  };

  if (!els.form || !els.input || !els.result) return;

  CLASS_KEYS.forEach((key) => {
    els.bars[key] = document.querySelector(`.bar[data-key="${key}"]`);
    els.pct[key] = $(`#pct-${key}`);
    els.fills[key] = $(`#fill-${key}`);
  });

  let model = null;
  let busy = false;
  let composing = false;

  /* ---------- 初始骨架 ---------- */
  initGauge();
  renderChips();
  bindEvents();
  bootModel();

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
  }

  /* ---------- 模型加载 ---------- */
  async function bootModel() {
    hideError();
    setLoading(true);
    setScanEnabled(false, '加载模型中…');
    try {
      model = await loadModel(MODEL_URL);
      renderModelCard();
      setLoading(false);
      setScanEnabled(true, '开始扫描');
      if (els.input) els.input.placeholder = '输入一个名字，如：林晚';
    } catch (err) {
      model = null;
      setLoading(false);
      setScanEnabled(false, '开始扫描');
      els.result.hidden = true;
      showError('模型加载失败', err && err.message ? err.message : String(err));
    }
  }

  function renderModelCard() {
    if (!model || !els.statNTrain) return;
    const data = modelCardData(model);
    els.statNTrain.textContent = data.nTrain == null ? '待训练' : formatInteger(data.nTrain);
    if (els.statTestAcc) {
      els.statTestAcc.textContent = data.testAcc3 == null ? '待训练' : formatPercent(data.testAcc3, 1);
    }
    if (els.statBinBalanced) {
      els.statBinBalanced.textContent =
        data.binaryBalancedAcc == null ? '待训练' : formatPercent(data.binaryBalancedAcc, 1);
    }
    if (els.statBaseline) els.statBaseline.textContent = formatBaseline(data.baseline);
    if (els.formatTag && data.format) els.formatTag.textContent = data.format;
    if (els.datasetLine) {
      els.datasetLine.textContent = data.hasDataset
        ? `训练数据：百合 ${formatInteger(data.perLabel[0])} · 纯爱 ${formatInteger(data.perLabel[1])} · 言情 ${formatInteger(data.perLabel[2])}`
        : '训练数据：待接入';
    }
  }

  /* ---------- 扫描 ---------- */
  async function handleScan() {
    if (busy) return;

    const raw = els.input.value;
    if (!raw || !raw.replace(/\s+/g, '')) {
      setHint('请先输入一个名字，比如「林晚」');
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
      const view = buildViewModel(model, raw);
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
    els.result.classList.remove('pop');
    // 强制重排，让入场动画每次都能重播
    void els.result.offsetWidth;
    els.result.classList.add('pop');

    // 仪表：百合指数 = 二分类头 P(百合)
    const ratio = Math.min(1, Math.max(0, view.baiheIndex));
    const radius = els.gaugeMeter.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;
    els.gaugeMeter.style.strokeDashoffset = `${circumference * (1 - ratio)}`;
    els.gauge.dataset.tier = view.gauge.tier;
    els.gauge.setAttribute('aria-label', `百合指数 ${view.baiheRounded}%`);
    els.gaugeNumber.textContent = String(view.baiheRounded);
    els.gaugeTier.textContent = view.gauge.label;

    // 点评 + 双头说明
    els.verdict.textContent = view.verdict;
    const decisionText = view.binaryDecision === 'baihe' ? '判定为「百合」' : '判定为「非百合」';
    els.leader.textContent =
      `二分类头：${decisionText}（P(百合)=${formatPercent(view.baiheIndex, 1)}）` +
      ` · 三分类头最像：${view.leader.name}（${view.leader.percent.toFixed(1)}%）`;

    // 三条概率条：三分类头（辅助参考）
    const topIndex = view.probs3.indexOf(Math.max(...view.probs3));
    CLASS_KEYS.forEach((key, index) => {
      const ratioValue = Math.min(1, Math.max(0, view.probs3[index]));
      if (els.pct[key]) els.pct[key].textContent = formatPercent(ratioValue, 1);
      if (els.fills[key]) els.fills[key].style.width = `${ratioValue * 100}%`;
      if (els.bars[key]) els.bars[key].classList.toggle('is-top', index === topIndex);
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

    // 逐字贡献（基于百合指数）
    els.contrib.replaceChildren();
    view.attribution.forEach((item) => {
      const points = item.delta * 100;
      const chip = document.createElement('span');
      chip.className = 'contrib-chip';
      if (points > 0.05) chip.classList.add('contrib-chip--pos');
      else if (points < -0.05) chip.classList.add('contrib-chip--neg');
      chip.title = `把「${item.char}」换成未知字后，百合指数 ${formatDelta(item.delta)}（单位：百分点）`;

      const charEl = document.createElement('span');
      charEl.className = 'contrib-chip__char';
      charEl.textContent = item.char;

      const deltaEl = document.createElement('span');
      deltaEl.className = 'contrib-chip__delta';
      deltaEl.textContent = formatDelta(item.delta);

      chip.append(charEl, deltaEl);
      els.contrib.append(chip);
    });

    // 移动端：结果在视口外时轻轻滚过去
    if (typeof window !== 'undefined' && els.result.getBoundingClientRect().top > window.innerHeight * 0.72) {
      els.result.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
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

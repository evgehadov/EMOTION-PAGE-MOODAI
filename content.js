// content.js — Emotion Page Analyzer (мгновенное обновление, индикатор прогресса)
// Финальная версия для защиты — работает быстро и понятно

(() => {
  'use strict';

  const API_URL = 'http://127.0.0.1:8000/predict_many';
  const BATCH_SIZE = 10;                  // уменьшили — обновления чаще
  const BUFFER_MULTIPLIER = 2;            // буфер вниз
  const WARNING_THRESHOLD = 0.20;
  const MIN_WORDS = 4;
  const TOOLTIP_DELAY = 2000;

 const EMOTION_COLOR_MAP = {
  // Позитивные
  'admiration': '#FFDDA0', 'восхищение': '#FFDDA0',
  'amusement': '#FF9966', 'веселье': '#FF9966', 'развлечение': '#FF9966',
  'approval': '#BEE7C0', 'одобрение': '#BEE7C0',
  'caring': '#BEE7C0', 'забота': '#BEE7C0',
  'curiosity': '#BEE7E3', 'любопытство': '#BEE7E3',
  'desire': '#FFC0D0', 'желание': '#FFC0D0',
  'excitement': '#FFD8B8', 'возбуждение': '#FFD8B8', 'волнение': '#FFD8B8',
  'gratitude': '#BEE7C0', 'благодарность': '#BEE7C0',
  'joy': '#FFDDA0', 'радость': '#FFDDA0',
  'love': '#FFC0D0', 'любовь': '#FFC0D0',
  'optimism': '#FFD8B8', 'оптимизм': '#FFD8B8',
  'pride': '#FFDDA0', 'гордость': '#FFDDA0',
  'relief': '#CDEBC5', 'облегчение': '#CDEBC5',

  // Негативные
  'anger': '#FFB3A3', 'гнев': '#FFB3A3', 'злость': '#FFB3A3',
  'annoyance': '#FFB3A3', 'раздражение': '#FFB3A3',
  'disappointment': '#B8C4E6', 'разочарование': '#B8C4E6',
  'disapproval': '#FFB3A3', 'неодобрение': '#FFB3A3',
  'disgust': '#D7B89E', 'отвращение': '#D7B89E',
  'embarrassment': '#B8C4E6', 'смущение': '#B8C4E6',
  'fear': '#C9A3FF', 'страх': '#C9A3FF',
  'grief': '#A7C7FF', 'горе': '#A7C7FF',
  'nervousness': '#C9A3FF', 'нервозность': '#C9A3FF',
  'remorse': '#B8C4E6', 'раскаяние': '#B8C4E6',
  'sadness': '#A7C7FF', 'печаль': '#336699', 'грусть': '#336699',

  // Амбивалентные
  'confusion': '#99CCFF', 'замешательство': '#99CCFF', 'непонимание': '#99CCFF',
  'realization': '#BEE7E3', 'осознание': '#BEE7E3',
  'surprise': '#BEE7E3', 'удивление': '#BEE7E3',

  // Нейтрально
  'neutral': '#999999', 'нейтрально': '#999999'
};

  const NEGATIVE_EMOTIONS = ['anger','sadness','fear','disgust','disappointment','злость','печаль','страх','отвращение','разочарование'];

  const PANEL_ID = 'emotion-analyzer-panel';
  const DETAIL_PANEL_ID = 'emotion-analyzer-detail';
  const TOOLTIP_ID = 'emotion-analyzer-tooltip';
  const STYLE_ID = 'emotion-analyzer-styles';

  let state = {
    analyzedPosts: new Map(),       // id -> данные поста
    accumulatedProbs: [],           // для среднего
    running: false,
    panelsInjected: false,
    detailOpen: false,
    tooltipTimer: null,
    scrollDebounce: null
  };

  /* ---------------------- УТИЛИТЫ ---------------------- */

  const sleep = ms => new Promise(r => setTimeout(r, ms));

  function safeText(node) { try { return node.innerText || node.textContent || ''; } catch (e) { return ''; } }

  function countWords(text) { return text.trim().split(/\s+/).filter(Boolean).length; }

  function containsCyrillic(text) { return /[а-яА-ЯЁё]/.test(text); }
  function containsLatin(text) { return /[A-Za-z]/.test(text); }
  function detectLang(text) { return containsCyrillic(text) ? 'ru' : containsLatin(text) ? 'en' : null; }

  function getColorForEmotion(key) {
    if (!key) return '#cccccc';
    return EMOTION_COLOR_MAP[key.toLowerCase()] || '#cccccc';
  }

  function looksLikeDate(text) {
    const t = text.trim();
    const datePatterns = [
      /\d{1,4}[-.\/]\d{1,2}[-.\/]\d{1,4}/,
      /\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4}/i
    ];
    if (datePatterns.some(p => p.test(t))) return true;
    const clean = t.replace(/[\d:\-\.\/\s]/g, '');
    return clean.length < t.length * 0.4;
  }

  function hasMediaChild(el) { return !!el.querySelector('audio, video, iframe, picture, img'); }

  function looksLikeAd(el) {
    const keywords = ['ad','ads','sponsored','promo','реклама','спонсор','спонсировано'];
    let cur = el;
    for (let i = 0; i < 5 && cur; i++) {
      const txt = (cur.className || cur.id || '' + safeText(cur)).toLowerCase();
      if (keywords.some(k => txt.includes(k))) return true;
      cur = cur.parentElement;
    }
    return false;
  }

  function isCollapsedElement(el) {
    if (el.tagName === 'DETAILS' && !el.hasAttribute('open')) return true;
    let cur = el;
    for (let i = 0; i < 6 && cur; i++) {
      if (cur.getAttribute?.('aria-expanded') === 'false') return true;
      cur = cur.parentElement;
    }
    return false;
  }

  function markPostElement(el) {
    if (!el.dataset.emotionPostId) el.dataset.emotionPostId = 'e' + Date.now() + Math.random();
    return el.dataset.emotionPostId;
  }

  function isInOrNearViewport(el) {
    const rect = el.getBoundingClientRect();
    const buffer = window.innerHeight * BUFFER_MULTIPLIER;
    return rect.top < window.innerHeight + buffer && rect.bottom > -buffer;
  }

  /* ---------------------- СБОР ПОСТОВ ---------------------- */

  function collectAllTextBlocks() {
    const selectors = ['article','section','div','p','li','blockquote','span','h1','h2','h3','h4','h5','h6'];
    const nodes = Array.from(document.querySelectorAll(selectors.join(',')));
    const candidates = [];
    const seen = new Set();

    for (const el of nodes) {
      const leafs = findLeafTextElements(el);
      for (const leaf of leafs) {
        const text = safeText(leaf).trim();
        if (!text || countWords(text) < MIN_WORDS || looksLikeDate(text) || looksLikeAd(leaf)) continue;
        if (looksLikeAd(leaf) || (hasMediaChild(leaf) && countWords(text) < 10) || isCollapsedElement(leaf)) continue;
        const lang = detectLang(text);
        if (!lang || !['ru','en'].includes(lang)) continue;

        const id = markPostElement(leaf);
        if (!seen.has(id) && isInOrNearViewport(leaf)) {
          seen.add(id);
          candidates.push({el: leaf, text, id});
        }
      }
    }
    return candidates;
  }

  function findLeafTextElements(root) {
    const leafs = [];
    const hasElementChildren = Array.from(root.childNodes || []).some(n => n.nodeType === 1);
    if (!hasElementChildren) return [root];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, {
      acceptNode: node => Array.from(node.childNodes || []).some(n => n.nodeType === 1) ? NodeFilter.FILTER_SKIP : NodeFilter.FILTER_ACCEPT
    });
    let n;
    while (n = walker.nextNode()) leafs.push(n);
    return leafs;
  }

  /* ---------------------- UI ---------------------- */

  function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `
      #${PANEL_ID}, #${DETAIL_PANEL_ID} { position: fixed; top: 10px; right: 12px; z-index: 2147483647; background: #fbf6ef; border: 1px solid rgba(0,0,0,0.06); border-radius: 10px; box-shadow: 0 8px 30px rgba(0,0,0,0.15); font-family: Inter, system-ui, sans-serif; padding: 12px; width: 340px; color: #111; }
      #${DETAIL_PANEL_ID} { top: 120px; background: #ffffff; max-height: 70vh; overflow-y: auto; }
      .emotion-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
      .emotion-main-box { padding: 8px 14px; border-radius: 8px; border: 1px solid; font-weight: 700; font-size: 18px; }
      .emotion-controls { display: flex; gap: 8px; justify-content: flex-end; margin-top: 10px; }
      .emotion-btn { padding: 6px 12px; border-radius: 8px; cursor: pointer; font-weight: 600; border: 1px solid rgba(0,0,0,0.06); background: #fff; }
      .emotion-btn.primary { background: #0066cc; color: #fff !important; border-color: rgba(0,0,0,0.06); }
.emotion-list-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 8px;
  border-bottom: 1px solid #f1f1f1;
  cursor: pointer;
}
      .emotion-list-item:hover {
  background: #fafafa;
}
    .emotion-preview {
    flex: 1;
    color: #000;
    font-weight: 500;
    font-size: 14px;
    margin-right: 10px;
    }
    .emotion-name {
    min-width: 100px;
    max-width: 130px;
    text-align: right;
    font-weight: 700;
    font-size: 17px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
   #${TOOLTIP_ID} { position: fixed; pointer-events: none; z-index: 2147483648; background: rgba(0,0,0,0.88); color: #fff; padding: 8px 10px; border-radius: 6px; font-size: 12px; max-width: 280px; white-space: pre-wrap; display: none; }
      .emotion-frame { transition: all 0.25s ease; box-sizing: border-box; }
      .emotion-frame.thin { border: 1.5px solid transparent; }
      .emotion-frame.warning { box-shadow: 0 0 0 2.5px rgba(255,100,100,0.22); }
    `;
    document.head.appendChild(style);
  }

  function createMainPanel() {
    if (document.getElementById(PANEL_ID)) return;
    injectStyles();
    const panel = document.createElement('div');
    panel.id = PANEL_ID;

    const row = document.createElement('div');
    row.className = 'emotion-row';
    const title = document.createElement('div');
    title.textContent = 'Эмоции страницы';
    title.style.fontWeight = '600';
    const mainBox = document.createElement('div');
    mainBox.id = 'emotion-main-box';
    mainBox.className = 'emotion-main-box';
    mainBox.textContent = 'Анализ...';
    row.appendChild(title);
    row.appendChild(mainBox);
    panel.appendChild(row);

    const controls = document.createElement('div');
    controls.className = 'emotion-controls';
    const refreshBtn = document.createElement('button');
    refreshBtn.className = 'emotion-btn';
    refreshBtn.textContent = 'Обновить';
    refreshBtn.onclick = () => {
      state.analyzedPosts.clear();
      state.accumulatedProbs = [];
      showMainEmotion('Анализ...', '#E6E6E6');
      analyzeNewPosts();
    };
    const detailBtn = document.createElement('button');
    detailBtn.className = 'emotion-btn primary';
    detailBtn.textContent = 'Подробнее';
    detailBtn.onclick = toggleDetailPanel;
    controls.appendChild(refreshBtn);
    controls.appendChild(detailBtn);
    panel.appendChild(controls);

    document.body.appendChild(panel);
    state.panelsInjected = true;
  }

function showMainEmotion(name, color) {
  const box = document.getElementById('emotion-main-box');
  if (!box) return;
  box.textContent = name;  // просто имя эмоции
  box.style.backgroundColor = color + '33';
  box.style.borderColor = color || '#ddd';
}

  /* ---------------------- АНАЛИЗ С МГНОВЕННЫМ ОБНОВЛЕНИЕМ ---------------------- */

  async function postPredictMany(texts) {
    try {
      const resp = await fetch(API_URL, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({texts}) });
      if (!resp.ok) return null;
      return await resp.json();
    } catch (e) {
      console.error(e);
      return null;
    }
  }

  async function analyzeNewPosts() {
    if (state.running) return;
    state.running = true;

    const candidates = collectAllTextBlocks();
    const newCandidates = candidates.filter(c => !state.analyzedPosts.has(c.id));

    if (newCandidates.length === 0) {
      state.running = false;
      return;
    }

    const texts = newCandidates.map(c => c.text);
    let offset = 0;

    while (offset < texts.length) {
      const batch = texts.slice(offset, offset + BATCH_SIZE);
      const resp = await postPredictMany(batch);

      const batchResults = resp || batch.map(() => null);

      newCandidates.slice(offset, offset + BATCH_SIZE).forEach((post, i) => {
        const result = batchResults[i];
        if (result?.probabilities) {
          const top = getTopEmotion(result.probabilities);
          const isWarning = isPostWarning(result.probabilities);
          state.analyzedPosts.set(post.id, {el: post.el, text: post.text, topKey: top.key, isWarning});
          state.accumulatedProbs.push(result.probabilities);

          post.el.dataset.emotionProb = JSON.stringify(result.probabilities);
          applyFrame(post.el, top.key, isWarning);
          attachHover(post.el);
        }
      });

      // МГНОВЕННОЕ ОБНОВЛЕНИЕ после каждого батча!
      updatePageEmotion();
      if (state.detailOpen) rebuildDetailPanel();

      offset += BATCH_SIZE;
      await sleep(80); // маленькая пауза, чтобы браузер не зависал
    }

    state.running = false;
  }

  function applyFrame(el, topKey, isWarning) {
    const color = getColorForEmotion(topKey);
    el.classList.add('emotion-frame', 'thin');
    el.style.borderColor = color;
    if (isWarning) el.classList.add('warning');
  }

  function attachHover(el) {
    // Чтобы не навешивать несколько раз
    el.removeEventListener('mouseenter', onMouseEnter);
    el.removeEventListener('mousemove', onMouseMove);
    el.removeEventListener('mouseleave', onMouseLeave);
    el.addEventListener('mouseenter', onMouseEnter);
    el.addEventListener('mousemove', onMouseMove);
    el.addEventListener('mouseleave', onMouseLeave);
  }

  function normalizeKey(key) { return key.toString().toLowerCase(); }

  function getTopEmotion(probObj, ignoreNeutral = false) {
    if (!probObj) return {key: 'neutral', value: 0};
    let entries = Object.entries(probObj).map(([k, v]) => [normalizeKey(k), v]);
    if (ignoreNeutral) entries = entries.filter(([k]) => !['neutral', 'нейтрально'].includes(k));
    if (entries.length === 0) return {key: 'neutral', value: 0};
    entries.sort((a, b) => b[1] - a[1]);
    return {key: entries[0][0], value: entries[0][1]};
  }

  function isPostWarning(probObj) {
    if (!probObj) return false;
    return Object.entries(probObj).some(([k, v]) => NEGATIVE_EMOTIONS.includes(normalizeKey(k)) && v >= WARNING_THRESHOLD);
  }

  function prettify(key) {
    if (!key) return '—';
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  function updatePageEmotion() {
    if (state.accumulatedProbs.length === 0) {
      showMainEmotion('Анализ...', '#E6E6E6');
      return;
    }

    const accum = {};
    state.accumulatedProbs.forEach(prob => {
      Object.entries(prob).forEach(([k, v]) => {
        const nk = normalizeKey(k);
        accum[nk] = (accum[nk] || 0) + v;
      });
    });

    const averaged = {};
    Object.keys(accum).forEach(k => averaged[k] = accum[k] / state.accumulatedProbs.length);

      // Сначала пытаемся найти эмоцию НЕ neutral
  let top = getTopEmotion(averaged, true);

  // Если ничего значимого не нашли — значит всё нейтрально
  if (top.key === 'neutral' || top.value < 0.05) {  // можно добавить порог, если хочешь
    top = getTopEmotion(averaged, false);  // берём реальный топ, включая neutral
  }

  const displayName = top.key === 'neutral' ? 'Нейтрально' : prettify(top.key);
  // или 'Спокойная', если хочешь мягче
  const color = getColorForEmotion(top.key);
  showMainEmotion(displayName, color);
  }

  /* ---------------------- TOOLTIP ---------------------- */

  function createTooltip() {
    let t = document.getElementById(TOOLTIP_ID);
    if (!t) {
      t = document.createElement('div');
      t.id = TOOLTIP_ID;
      document.body.appendChild(t);
    }
    return t;
  }

  function onMouseEnter(e) {
    const el = e.currentTarget;
    if (!el.dataset.emotionProb) return;
    state.tooltipTimer = setTimeout(() => {
      try {
        const prob = JSON.parse(el.dataset.emotionProb);
        const pairs = Object.entries(prob).map(([k, v]) => [k.toLowerCase(), v]).sort((a, b) => b[1] - a[1]);
        const lines = pairs.map(([k, v]) => `${prettify(k)}: ${(v * 100).toFixed(1)}%`);
        const tooltip = createTooltip();
        tooltip.textContent = lines.join('\n');
        tooltip.style.display = 'block';
        positionTooltip(e, tooltip);
      } catch (_) {}
    }, TOOLTIP_DELAY);
  }

  function onMouseMove(e) {
    const t = document.getElementById(TOOLTIP_ID);
    if (t?.style.display === 'block') positionTooltip(e, t);
  }

  function onMouseLeave() {
    clearTimeout(state.tooltipTimer);
    const t = document.getElementById(TOOLTIP_ID);
    if (t) t.style.display = 'none';
  }

  function positionTooltip(e, tooltip) {
    let x = e.clientX + 15;
    let y = e.clientY + 10;
    const rect = tooltip.getBoundingClientRect();
    if (x + rect.width > window.innerWidth) x = e.clientX - rect.width - 15;
    if (y + rect.height > window.innerHeight) y = e.clientY - rect.height - 15;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
  }

  /* ---------------------- ДЕТАЛЬНАЯ ПАНЕЛЬ ---------------------- */

  function toggleDetailPanel() {
    if (state.detailOpen) {
      document.getElementById(DETAIL_PANEL_ID)?.remove();
      state.detailOpen = false;
    } else {
      state.detailOpen = true;
      buildDetailPanel();
    }
  }

  function rebuildDetailPanel() {
    document.getElementById(DETAIL_PANEL_ID)?.remove();
    if (state.detailOpen) buildDetailPanel();
  }

  function buildDetailPanel() {
    const panel = document.createElement('div');
    panel.id = DETAIL_PANEL_ID;

    const header = document.createElement('div');
    header.style.display = 'flex';
    header.style.justifyContent = 'space-between';
    header.style.marginBottom = '8px';
    const h = document.createElement('div');
    h.textContent = `Посты (${state.analyzedPosts.size})`;
    h.style.fontWeight = '600';
    const close = document.createElement('button');
    close.textContent = '×';
    close.style.fontSize = '20px';
    close.style.cursor = 'pointer';
    close.onclick = () => { panel.remove(); state.detailOpen = false; };
    header.appendChild(h);
    header.appendChild(close);
    panel.appendChild(header);

    const list = document.createElement('div');
    Array.from(state.analyzedPosts.values()).sort((a, b) => {
      // сортировка по позиции на странице (сверху вниз)
      return a.el.getBoundingClientRect().top - b.el.getBoundingClientRect().top;
    }).forEach(post => {
      const item = document.createElement('div');
      item.className = 'emotion-list-item';
      const preview = document.createElement('div');
      preview.className = 'emotion-preview';
      preview.textContent = post.text.trim().split(/\s+/).slice(0, 3).join(' ') + '...';
      const emo = document.createElement('div');
      emo.className = 'emotion-name';
      emo.textContent = prettify(post.topKey);
      emo.style.color = getColorForEmotion(post.topKey);
      item.appendChild(preview);
      item.appendChild(emo);
      item.onclick = () => post.el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      list.appendChild(item);
    });
    panel.appendChild(list);
    document.body.appendChild(panel);
  }

  /* ---------------------- СОБЫТИЯ ---------------------- */

  function triggerAnalysis() {
    if (state.scrollDebounce) clearTimeout(state.scrollDebounce);
    state.scrollDebounce = setTimeout(analyzeNewPosts, 500);
  }

  function init() {
    createMainPanel();
    showMainEmotion('Анализ...', '#E6E6E6');
    analyzeNewPosts();

    window.addEventListener('scroll', triggerAnalysis);
    window.addEventListener('resize', triggerAnalysis);

    const observer = new MutationObserver(triggerAnalysis);
    observer.observe(document.body, { childList: true, subtree: true, attributes: true });

    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) analyzeNewPosts();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
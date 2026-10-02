"""Render the CloudWatch-style dashboard mockups from the committed site data.

    uv run python docs/mockups/build_mockups.py   ->  docs/mockups/dashboard.html

One file; the header's theme menu switches light / dark / system (default: follows the OS).

Each mockup is one self-contained HTML file (data inlined); the "Modal runs" widget shows
placeholder jobs, marked MOCK, because no live status source exists yet.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "site/public/data/rows.json").read_text())
TLDR = (ROOT / "data/tldr.md").read_text()

MOCK_QUEUE = [
    {"label": "kev9b_v2_noul", "source": "huggingface", "status": "running", "min": 12},
    {"label": "kev2b_per2021", "source": "huggingface", "status": "running", "min": 3},
    {"label": "laya421m_typed_reworded", "source": "submitted", "status": "queued", "min": 95},
    {"label": "issue_4_mxbai_rerank", "source": "issue", "status": "queued", "min": 1440},
]

THEMES = {
    "light": {
        "bg": "#f2f3f3",
        "panel": "#ffffff",
        "border": "#d5dbdb",
        "text": "#16191f",
        "muted": "#687078",
        "head": "#232f3e",
        "headtext": "#ffffff",
        "accent": "#0972d3",
        "grid": "#eaeded",
        "hl": "#fff7e0",
        "good": "#1d8102",
        "bad": "#d13212",
        "chip": "#f1faff",
        "chipborder": "#8fb8e0",
    },
    "dark": {
        "bg": "#0f1b2a",
        "panel": "#161e2d",
        "border": "#2a3650",
        "text": "#d1d5db",
        "muted": "#8d99ae",
        "head": "#0b1320",
        "headtext": "#f3f4f6",
        "accent": "#539fe5",
        "grid": "#223049",
        "hl": "#2a2b1c",
        "good": "#29ad32",
        "bad": "#ff7a66",
        "chip": "#1b2a44",
        "chipborder": "#3b5a8a",
    },
}

CSS = """
%(vars)s
select.theme { background: transparent; color: #cfd8e3; border: 1px solid #4b5563; border-radius: 4px; padding: 2px 6px; font: inherit; font-size: 12px; }
select.theme option { color: #16191f; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font: 14px/1.45 "Amazon Ember", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
.topbar { background:var(--head); color:var(--headtext); height:48px; display:flex; align-items:center; padding:0 20px; gap:18px; }
.topbar .brand { font-weight:700; font-size:16px; letter-spacing:.2px; }
.topbar .crumb { color:#aab7c4; font-size:13px; }
.topbar .right { margin-left:auto; display:flex; gap:10px; align-items:center; font-size:13px; color:#cfd8e3; }
.topbar .pill { border:1px solid #4b5a6a; border-radius:14px; padding:2px 10px; }
.toolbar { display:flex; align-items:center; gap:10px; padding:10px 20px 0; flex-wrap:wrap; }
.toolbar .label { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin-right:2px; }
.chip { border:1px solid var(--border); background:var(--panel); border-radius:14px; padding:2px 10px; font-size:12px; cursor:pointer; user-select:none; color:var(--text); }
.chip.on { background:var(--chip); border-color:var(--chipborder); color:var(--accent); font-weight:600; }
.bar i.busy { width:35%%; animation: slide 1.6s ease-in-out infinite alternate; }
@keyframes slide { from { margin-left:0 } to { margin-left:65%% } }
.chip.reset { border-style:dashed; color:var(--muted); }
.dd { position:relative; }
.dd > button { border:1px solid var(--border); background:var(--panel); color:var(--text); border-radius:6px; padding:5px 10px; font-size:13px; cursor:pointer; min-width:150px; text-align:left; display:flex; justify-content:space-between; gap:10px; }
.dd > button .cnt { color:var(--accent); font-weight:600; }
.dd .menu { display:none; position:absolute; z-index:5; top:34px; left:0; background:var(--panel); border:1px solid var(--border); border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,.18); padding:6px 0; min-width:260px; max-height:360px; overflow:auto; }
.dd.open .menu { display:block; }
.dd .menu label { display:flex; align-items:center; gap:8px; padding:5px 12px; font-size:13px; cursor:pointer; white-space:nowrap; }
.dd .menu label:hover { background:var(--bg); }
.dd .menu .all { color:var(--muted); font-size:12px; padding:4px 12px 6px; border-bottom:1px solid var(--grid); margin-bottom:4px; display:flex; gap:12px; }
.dd .menu .all a { cursor:pointer; }
.tabs { display:flex; gap:2px; padding:10px 20px 0; border-bottom:1px solid var(--border); margin:0 0 14px; }
.tabs button { background:none; border:0; border-bottom:3px solid transparent; color:var(--muted); padding:8px 12px; font-size:14px; cursor:pointer; }
.tabs button.on { color:var(--text); border-bottom-color:var(--accent); font-weight:600; }
.grid { display:grid; grid-template-columns: repeat(12, 1fr); gap:14px; padding:0 20px 24px; }
.w { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:12px 14px; min-width:0; display:flex; flex-direction:column; }
.w h2 { margin:0 0 8px; font-size:13px; font-weight:700; color:var(--text); display:flex; align-items:center; gap:8px; }
.w h2 .sub { font-weight:400; color:var(--muted); }
.w h2 .menu { margin-left:auto; color:var(--muted); font-weight:400; letter-spacing:2px; cursor:default; }
.s3 { grid-column: span 3; } .s4 { grid-column: span 4; } .s6 { grid-column: span 6; } .s8 { grid-column: span 8; } .s12 { grid-column: span 12; }
.num .v { font-size:34px; font-weight:700; line-height:1.1; font-variant-numeric: tabular-nums; }
.num .who { margin-top:4px; font-weight:600; }
.num .cmp { color:var(--muted); font-size:12px; }
.num .delta { font-size:12px; font-weight:600; margin-left:6px; }
.good { color:var(--good); } .bad { color:var(--bad); }
.status { display:flex; gap:12px; flex-wrap:wrap; align-items:stretch; }
.job { flex:1 1 220px; border:1px solid var(--border); border-radius:6px; padding:8px 10px; background:var(--bg); }
.job .m { font-weight:600; } .job .meta { color:var(--muted); font-size:12px; }
.bar { height:6px; background:var(--grid); border-radius:3px; margin-top:6px; overflow:hidden; }
.bar i { display:block; height:100%%; background:var(--accent); }
.job.queued .bar i { background:var(--muted); width:0; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%%; margin-right:6px; background:var(--good); box-shadow:0 0 0 3px rgba(29,129,2,.18); }
.queued .dot { background:var(--muted); box-shadow:none; }
.mock { font-size:10px; border:1px solid var(--bad); color:var(--bad); border-radius:3px; padding:0 4px; letter-spacing:.08em; }
svg text { fill: var(--muted); font-size:11px; }
svg .axis { stroke: var(--border); } svg .gridline { stroke: var(--grid); }
.legend { display:flex; flex-wrap:wrap; gap:4px 12px; font-size:12px; margin-top:6px; }
.legend span { cursor:pointer; } .legend span.off { opacity:.35; }
.tldr p { margin:0 0 8px; } .tldr { font-size:13px; }
table { border-collapse:collapse; width:100%%; font-size:13px; }
th, td { padding:5px 8px; border-bottom:1px solid var(--grid); text-align:left; white-space:nowrap; }
th { color:var(--muted); font-weight:600; cursor:pointer; user-select:none; position:sticky; top:0; background:var(--panel); }
th.n, td.n { text-align:right; font-variant-numeric: tabular-nums; }
tr.hl td { background: var(--hl); }
td.dim { color: var(--muted); }
.tablewrap { overflow:auto; max-height:520px; }
.hint { color:var(--muted); font-size:12px; margin:0 0 8px; }
.eval input { width:100%%; padding:9px 10px; border:1px solid var(--border); border-radius:6px; background:var(--bg); color:var(--text); font-size:14px; }
.eval button { margin-top:10px; background:var(--accent); color:#fff; border:0; border-radius:6px; padding:8px 16px; font-weight:600; cursor:pointer; }
.hbar { display:grid; grid-template-columns: 170px 1fr 64px; gap:6px 10px; align-items:center; font-size:12px; }
.hbar .n { text-align:right; font-variant-numeric: tabular-nums; }
.hbar .t { height:10px; background:var(--grid); border-radius:3px; overflow:hidden; } .hbar .t i { display:block; height:100%%; }
.hbar .lbl { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.foot { padding:0 20px 24px; color:var(--muted); font-size:12px; }
"""

JS = r"""
const D = DATA; const KS = D.ks; const ROWS = D.rows.filter(r => r.queries === 75);
// Okabe-Ito colorblind-safe palette; each family also gets its own marker shape and dash so colour is never the only cue.
const PALETTE = ['#0072b2','#e69f00','#009e73','#d55e00','#56b4e9','#cc79a7','#b8a400'];
const MARKERS = ['circle', 'square', 'triangle', 'diamond'];
const REF_STYLE = {
  jev: {stroke: 'var(--text)', dash: '', width: 2.5, marker: 'circle'},
  production: {stroke: 'var(--text)', dash: '7 4', width: 2, marker: 'square'},
  random: {stroke: 'var(--muted)', dash: '2 4', width: 2, marker: 'triangle'},
  oracle: {stroke: 'var(--muted)', dash: '1 5', width: 1.5, marker: 'diamond'},
};
const fam = r => r.family;
const sk = r => r.serving.split(':')[0];
const FAMS = [...new Set(ROWS.map(fam))];
const GPUS = [...new Set(ROWS.map(r => r.gpu))];
const color = Object.fromEntries(FAMS.map((f, i) => [f, PALETTE[i % PALETTE.length]]));
// Styles are assigned per chart, in rank order of the plotted series, so the ≤7 non-reference lines never share a colour.
const chartStyles = S => Object.fromEntries(S.filter(r => !REF_STYLE[fam(r)]).map((r, i) => [fam(r), {
  stroke: PALETTE[i % PALETTE.length], dash: i >= PALETTE.length ? '9 3' : '', width: 1.8, marker: MARKERS[i % MARKERS.length]
}]));
function marker(shape, cx, cy, fill, r = 3.5) {
  if (shape === 'square') return `<rect x="${cx - r}" y="${cy - r}" width="${2 * r}" height="${2 * r}" fill="${fill}"/>`;
  if (shape === 'triangle') return `<polygon points="${cx},${cy - r - 1} ${cx - r - 1},${cy + r} ${cx + r + 1},${cy + r}" fill="${fill}"/>`;
  if (shape === 'diamond') return `<polygon points="${cx},${cy - r - 1} ${cx + r + 1},${cy} ${cx},${cy + r + 1} ${cx - r - 1},${cy}" fill="${fill}"/>`;
  return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}"/>`;
}
const swatch = (f, styles) => { const st = REF_STYLE[f] || styles[f]; return `<svg width="30" height="12" style="vertical-align:middle;margin-right:5px"><line x1="0" x2="30" y1="6" y2="6" stroke="${st.stroke}" stroke-width="${st.width}" ${st.dash ? `stroke-dasharray="${st.dash}"` : ''}/>${marker(st.marker, 15, 6, st.stroke, 3)}</svg>`; };
const state = { tab: 'summary', fams: new Set(FAMS), gpus: new Set(GPUS), sort: null, hidden: new Set() };
const fmt = (v, d) => v == null ? '–' : v.toLocaleString('en-US', {minimumFractionDigits: d, maximumFractionDigits: d});
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pass = r => state.fams.has(fam(r)) && state.gpus.has(r.gpu);
const visible = () => ROWS.filter(pass);

function dropdown(el, name, opts, set) {
  const open = el.classList.contains('open');
  el.innerHTML = `<button type="button">${name}${set.size === opts.length ? ` <span style="color:var(--muted)">all ${opts.length}</span>` : ` <span class="cnt">${set.size} of ${opts.length}</span>`}<span>▾</span></button><div class="menu"><div class="all"><a data-a="all">select all</a><a data-a="none">clear</a></div>${opts.map(o => `<label><input type="checkbox" ${set.has(o) ? 'checked' : ''} data-v="${esc(o)}">${esc(o)}</label>`).join('')}</div>`;
  el.classList.toggle('open', open);
  el.querySelector('button').onclick = e => { e.stopPropagation(); document.querySelectorAll('.dd').forEach(d => d !== el && d.classList.remove('open')); el.classList.toggle('open'); };
  el.querySelector('.menu').onclick = e => e.stopPropagation();
  el.querySelectorAll('input').forEach(c => c.onchange = () => { const v = c.dataset.v; set.has(v) ? set.delete(v) : set.add(v); render(); });
  el.querySelectorAll('.all a').forEach(a => a.onclick = () => { set.clear(); if (a.dataset.a === 'all') opts.forEach(o => set.add(o)); render(); });
}
document.addEventListener('click', () => document.querySelectorAll('.dd.open').forEach(d => d.classList.remove('open')));

// One line per model family: its best run by mean kept-mass among the visible rows.
function series(rows) {
  const best = {};
  for (const r of rows) if (r.kept_mass && r.mean_kept_mass != null) {
    const f = fam(r); if (!best[f] || r.mean_kept_mass > best[f].mean_kept_mass) best[f] = r;
  }
  const all = Object.values(best).sort((a, b) => b.mean_kept_mass - a.mean_kept_mass);
  const refs = all.filter(r => REFS.has(fam(r)));
  const top = all.filter(r => !REFS.has(fam(r))).slice(0, TOP_N - refs.length);
  return all.filter(r => refs.includes(r) || top.includes(r));
}
const TOP_N = 10, REFS = new Set(['jev', 'production', 'random']);

function lineChart(rows, height) {
  const S = series(rows); const styles = chartStyles(S); const W = 900, H = height, L = 48, R = 36, T = 12, B = 30;
  const ys = S.flatMap(r => KS.map(k => r.kept_mass[k]));
  const y0 = Math.max(0, Math.floor(Math.min(...ys) * 10) / 10 - 0.05), y1 = 1.0;
  const x = i => L + i * (W - L - R) / (KS.length - 1), y = v => T + (H - T - B) * (1 - (v - y0) / (y1 - y0));
  let g = '';
  for (let v = Math.ceil(y0 * 10) / 10; v <= y1 + 1e-9; v += 0.1) g += `<line class="gridline" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(1)}</text>`;
  KS.forEach((k, i) => g += `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">k=${k}</text>`);
  for (const r of S) {
    if (state.hidden.has(fam(r))) continue;
    const pts = KS.map((k, i) => `${x(i)},${y(r.kept_mass[k])}`).join(' ');
    const st = REF_STYLE[fam(r)] || styles[fam(r)];
    g += `<polyline fill="none" stroke="${st.stroke}" stroke-width="${st.width}" ${st.dash ? `stroke-dasharray="${st.dash}"` : ''} points="${pts}"><title>${esc(r.label)} · ${esc(r.serving)}</title></polyline>`;
    KS.forEach((k, i) => g += `<g>${marker(st.marker, x(i), y(r.kept_mass[k]), st.stroke)}<title>${esc(r.label)} @${k}: ${r.kept_mass[k].toFixed(3)}</title></g>`);
  }
  const legend = S.map(r => `<span class="${state.hidden.has(fam(r)) ? 'off' : ''}" data-f="${esc(fam(r))}">${swatch(fam(r), styles)}${esc(fam(r))} <span style="color:var(--muted)">${r.mean_kept_mass.toFixed(3)}</span></span>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="none" style="height:${H}px">${g}<line class="axis" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}"/><line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/></svg><div class="legend">${legend}</div>`;
}

function hbars(rows, get, digits, lower_is_better) {
  const S = {};
  for (const r of rows) { const v = get(r); if (v == null || v === 0) continue; const f = fam(r); if (!S[f] || v < S[f].v) S[f] = {v, r}; }
  const arr = Object.values(S).sort((a, b) => a.v - b.v); const max = Math.max(...arr.map(a => a.v));
  const jev = arr.find(a => fam(a.r) === 'jev');
  return `<div class="hbar">${arr.map(a => `<span class="lbl" title="${esc(a.r.label)} · ${esc(a.r.serving)} · ${esc(a.r.gpu)}">${esc(fam(a.r))} <span style="color:var(--muted)">${esc(a.r.gpu)}</span></span><span class="t"><i style="width:${100 * a.v / max}%;background:${color[fam(a.r)]}"></i></span><span class="n ${jev && a !== jev ? (a.v < jev.v === lower_is_better ? 'good' : '') : ''}">${fmt(a.v, digits)}</span>`).join('')}</div>`;
}

const COLS = {
  summary: [...KS.map(k => ['@' + k, r => r.kept_mass?.[k], 3]), ['mean', r => r.mean_kept_mass, 3], ['$ / 1k q', r => r.cost?.usd_per_1k, 1], ['s / query', r => r.latency?.s_per_query, 2]],
  quality: KS.map(k => ['kept-mass @' + k, r => r.kept_mass?.[k], 3]),
  cost: [['warm GPU-s', r => r.cost?.warm_gpu_s, 0], ['load s', r => r.cost?.load_s, 0], ['$ / run', r => r.cost?.warm_usd, 2], ['$ / run + load', r => r.cost?.in_function_usd, 2], ['$ / query', r => r.cost?.usd_per_query, 4], ['$ / 1k queries', r => r.cost?.usd_per_1k, 1], ['peak GB', r => r.cost?.peak_gb, 1]],
  latency: [['run (s)', r => r.latency?.run_s, 0], ['s / query', r => r.latency?.s_per_query, 2], ['hours / 1k', r => r.latency?.h_per_1k, 2]],
};
const LABELS = [['ranker', r => r.label], ['serving', r => r.serving], ['GPU', r => r.gpu]];

function table(rows, cols) {
  const s = state.sort; let sorted = rows;
  if (s) { const lab = LABELS.find(l => l[0] === s.key)?.[1]; const col = cols.find(c => c[0] === s.key);
    sorted = [...rows].sort((a, b) => lab ? lab(a).localeCompare(lab(b)) * s.dir : (((col[1](a) ?? -Infinity) - (col[1](b) ?? -Infinity)) * s.dir)); }
  const ar = k => s?.key === k ? (s.dir === 1 ? ' ▲' : ' ▼') : '';
  return `<p class="hint">Click a column header to sort; click again to reverse.</p><div class="tablewrap"><table><thead><tr>${LABELS.map(l => `<th data-k="${l[0]}">${l[0]}${ar(l[0])}</th>`).join('')}${cols.map(c => `<th class="n" data-k="${esc(c[0])}">${esc(c[0])}${ar(c[0])}</th>`).join('')}</tr></thead><tbody>${
    sorted.map(r => `<tr class="${r.highlight ? 'hl' : ''}">${LABELS.map(l => `<td title="${esc(r.experiment)}">${esc(l[1](r))}</td>`).join('')}${cols.map(c => { const v = c[1](r); return v == null ? '<td class="n dim">–</td>' : `<td class="n">${fmt(v, c[2])}</td>`; }).join('')}</tr>`).join('')}</tbody></table></div>`;
}

const widget = (span, title, body, sub = '', extra = '') => `<section class="w s${span} ${extra}"><h2>${title}${sub ? ` <span class="sub">${sub}</span>` : ''}<span class="menu">⋮</span></h2>${body}</section>`;

function numWidget(card, better) {
  const [who, jev] = card.detail.split(' · Jev ');
  const v = parseFloat(card.value), j = parseFloat(jev); const rel = (v - j) / j;
  const good = better === 'high' ? v >= j : v <= j;
  return widget(4, card.label.split(' (')[0], `<div class="num"><div class="v">${card.value}<span class="delta ${good ? 'good' : 'bad'}">${rel >= 0 ? '+' : ''}${(100 * rel).toFixed(0)}% vs Jev</span></div><div class="who">${esc(who)}</div><div class="cmp">${card.label.match(/\((.*)\)/)[1]} · Jev ${jev}</div></div>`);
}

function queueWidget() {
  const age = m => m < 60 ? `${m} min` : m < 1440 ? `${Math.round(m / 60)} h` : `${Math.round(m / 1440)} d`;
  const items = QUEUE.map(j => `<div class="job ${j.status}"><div class="m"><span class="dot"></span>${esc(j.label)}</div>
    <div class="meta">${j.status === 'running' ? `running · started ${age(j.min)} ago` : `queued · waiting ${age(j.min)}`} · from ${esc(j.source)}</div>
    ${j.status === 'running' ? '<div class="bar"><i class="busy"></i></div>' : ''}</div>`).join('');
  const running = QUEUE.filter(j => j.status === 'running').length;
  return widget(12, `Model evaluation queue <span class="sub">${running} running · ${QUEUE.length - running} waiting</span>`,
    `<div class="status">${items || '<div class="meta">queue is empty</div>'}</div>`,
    '<span class="mock">MOCK</span> source: data/queue.json on main · several can run at once · a finished model leaves the queue and appears in the Model list');
}

function render() {
  dropdown($('#fams'), 'Model', FAMS, state.fams); dropdown($('#gpus'), 'GPU', GPUS, state.gpus);
  $('#shown').textContent = `${visible().length} of ${ROWS.length} runs`;
  document.querySelectorAll('.tabs button').forEach(b => b.classList.toggle('on', b.dataset.t === state.tab));
  const rows = visible(); let g = '';
  if (state.tab === 'summary') {
    g += queueWidget();
    g += numWidget(D.cards[0], 'high') + numWidget(D.cards[1], 'low') + numWidget(D.cards[2], 'low');
    g += widget(8, 'Quality · kept-mass@k', lineChart(rows, 300), 'Top 10 shown (best run per model, Jev / production / random always included) · full list in the table · click a legend entry to hide it');
    g += widget(4, 'TLDR', `<div class="tldr">${D.tldr.split('\n').filter(l => l.trim()).map(l => `<p>${esc(l)}</p>`).join('')}</div>`, 'written by the daily run');
    g += widget(6, 'Cost · $ per 1k queries', hbars(rows, r => r.cost?.usd_per_1k, 2, true), 'cheapest run per family');
    g += widget(6, 'Latency · seconds per query', hbars(rows, r => r.latency?.s_per_query, 2, true), 'fastest run per family');
    g += widget(12, 'All runs', table(rows, COLS.summary), `${rows.length} rows · 75 frozen queries`);
  } else if (state.tab === 'quality') {
    g += widget(12, 'Quality · kept-mass@k', lineChart(rows, 360), 'Top 10 shown (best run per model, Jev / production / random always included) · full list in the table below');
    g += widget(12, 'Quality · all runs', table(rows.filter(r => r.tables.includes('quality')), COLS.quality));
  } else if (state.tab === 'cost') {
    g += widget(12, 'Cost · $ per 1k queries', hbars(rows, r => r.cost?.usd_per_1k, 2, true));
    g += widget(12, 'Cost · all runs', table(rows.filter(r => r.tables.includes('cost')), COLS.cost));
  } else if (state.tab === 'latency') {
    g += widget(12, 'Latency · seconds per query', hbars(rows, r => r.latency?.s_per_query, 2, true));
    g += widget(12, 'Latency · all runs', table(rows.filter(r => r.tables.includes('latency')), COLS.latency));
  } else {
    g += widget(6, 'Evaluate a new model', `<div class="eval"><p class="hint">Paste one link (Hugging Face model, GitHub repo, paper, API page) or describe the model. It is saved under <a href="#">requests/</a>; the next daily run picks it up and Devin works out how to run it.</p><input type="text" placeholder="text here:"><button>Submit</button></div>`);
    g += widget(6, 'Recent submissions', `<div class="tldr"><p>2026-10-02 04:50 UTC · <span style="color:var(--muted)">test submission from Devin via the Render site; ignore</span></p><p class="hint">Submissions appear here with their triage verdict once the daily run has looked at them.</p></div>`, '<span class="mock">MOCK</span>');
  }
  $('#grid').innerHTML = g;
  document.querySelectorAll('.legend span[data-f]').forEach(s => s.onclick = () => { const f = s.dataset.f; state.hidden.has(f) ? state.hidden.delete(f) : state.hidden.add(f); render(); });
  document.querySelectorAll('th[data-k]').forEach(h => h.onclick = () => { const k = h.dataset.k; state.sort = state.sort?.key === k ? (state.sort.dir === 1 ? {key: k, dir: -1} : null) : {key: k, dir: 1}; render(); });
}
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => { state.tab = b.dataset.t; state.sort = null; render(); });
$('#reset').onclick = () => { state.fams = new Set(FAMS); state.gpus = new Set(GPUS); state.hidden.clear(); state.sort = null; render(); };
render();
"""

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>jev-tracker · dashboard mockup</title><style>%(css)s</style></head>
<body>
<header class="topbar"><span class="brand">jev-tracker</span><span class="crumb">Dashboards › Jev alternatives · 75 frozen queries</span>
  <div class="right"><span>Last updated %(updated)s</span><span class="pill">↻ 1 min</span><select class="theme" id="theme" title="theme"><option value="system">system</option><option value="light">light</option><option value="dark">dark</option></select><a style="color:#cfd8e3" href="https://github.com/marcus-rox/jev-tracker">GitHub</a></div></header>
<div class="toolbar"><div class="dd" id="fams"></div><div class="dd" id="gpus"></div><span class="chip reset" id="reset">reset</span><span class="label" id="shown"></span></div>
<nav class="tabs"><button data-t="summary">Summary</button><button data-t="quality">Quality</button><button data-t="cost">Cost</button><button data-t="latency">Latency</button><button data-t="evaluate">Evaluate a new model</button></nav>
<main class="grid" id="grid"></main>
<p class="foot">Mockup rendered from site/public/data/rows.json (%(n)d rows, %(ne)d experiments). Widgets marked MOCK use placeholder data.</p>
<script>const DATA = %(data)s; const QUEUE = %(runs)s;</script>
<script>%(js)s</script>
</body></html>
"""

VARS = """
:root, [data-theme=light] { %(light)s }
[data-theme=dark] { %(dark)s }
@media (prefers-color-scheme: dark) { :root:not([data-theme=light]) { %(dark)s } }
"""

THEME_JS = """
const themeSel = document.getElementById('theme');
const applyTheme = v => {
  if (v === 'system') delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = v;
  localStorage.setItem('theme', v);
  themeSel.value = v;
};
applyTheme(localStorage.getItem('theme') || 'system');
themeSel.onchange = () => applyTheme(themeSel.value);
"""


def css_vars(colors: dict[str, str]) -> str:
    return " ".join(f"--{name}:{value};" for name, value in colors.items())


html = HTML % {
    "css": CSS % {"vars": VARS % {name: css_vars(colors) for name, colors in THEMES.items()}},
    "updated": DATA["updated"],
    "n": len(DATA["rows"]),
    "ne": len(DATA["experiments"]),
    "data": json.dumps({**DATA, "tldr": TLDR}),
    "runs": json.dumps(MOCK_QUEUE),
    "js": JS + THEME_JS,
}
(OUT / "dashboard.html").write_text(html)
print(f"wrote docs/mockups/dashboard.html ({len(html) // 1024} KB)")

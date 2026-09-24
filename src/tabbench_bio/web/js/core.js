/* Core globals, per-tab UI state, and small shared helpers.
 * Loaded first; every other module reads `DATA` / `COLORS` / `state` from here.
 */

let DATA = null;
let COLORS = {};
const state = {};

function panel() { return document.getElementById("panel"); }

function colVisible(c, st) { return c.fixed || !st.hidden.has(c.key); }

function initState() {
  for (const t of DATA.order) {
    const tab = DATA.tabs[t];
    if (tab.kind === "breakdown") {
      state[t] = { axisKey: (tab.axes[0] || {}).key }; // which profile axis to group by
      continue;
    }
    const hidden = new Set();
    for (const c of tab.columns) { if (!c.fixed && c.default === false) hidden.add(c.key); }
    const filters = {};
    for (const f of (tab.filters || [])) filters[f.column] = new Set(); // excluded values
    state[t] = { sort: { key: (tab.columns[0] || {}).key, dir: "asc" }, hidden, filters };
  }
}

function distinct(rows, key) {
  const s = new Set();
  for (const r of rows) { if (r[key] !== null && r[key] !== undefined) s.add(r[key]); }
  return [...s].sort();
}

function passesFilters(row, st, tab) {
  for (const f of (tab.filters || [])) {
    const ex = st.filters[f.column];
    if (ex.size && ex.has(row[f.column])) return false;
  }
  return true;
}

function fmt(c, v) {
  if (v === null || v === undefined || (typeof v === "number" && isNaN(v))) return "—";
  if (c.type === "num") return Number(v).toFixed(c.digits);
  if (c.type === "int") return String(Math.round(Number(v)));
  return String(v);
}

function bestValues(cols, rows) {
  const best = {};
  for (const c of cols) {
    if ((c.type !== "num" && c.type !== "int") || !c.better) continue;
    let b = null;
    for (const r of rows) {
      const v = r[c.key];
      if (v === null || v === undefined || isNaN(v)) continue;
      b = (b === null) ? v : (c.better === "high" ? Math.max(b, v) : Math.min(b, v));
    }
    if (b !== null) best[c.key] = b;
  }
  return best;
}

function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }

function selectControl(label, options, value, onChange, labelFn) {
  const wrap = document.createElement("label");
  wrap.className = "bd-axis";
  const lab = document.createElement("span");
  lab.className = "bd-axis-label";
  lab.textContent = label;
  const sel = document.createElement("select");
  for (const o of options) {
    const opt = document.createElement("option");
    opt.value = o;
    opt.textContent = labelFn ? labelFn(o) : o;
    sel.appendChild(opt);
  }
  sel.value = value;
  sel.onchange = () => onChange(sel.value);
  wrap.append(lab, sel);
  return wrap;
}

function kfmt(n) { n = Number(n); return n >= 1000 && n % 1000 === 0 ? n / 1000 + "k" : String(n); }
function sampleLabel(s) { return String(s).toLowerCase() === "full" ? "Full" : String(s); }
function capLabel(c) { return String(c).toLowerCase() === "full" ? "Full" : kfmt(c); }
function domainLabel(d) { return String(d) === "all" ? "All" : String(d); }
const METRIC_LABELS = { balanced_accuracy: "Balanced Accuracy", matthews_corrcoef: "MCC", roc_auc: "ROC-AUC", f1_macro: "Macro-F1", f1_score: "F1" };
function metricLabel(m) { return METRIC_LABELS[m] || String(m).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()); }

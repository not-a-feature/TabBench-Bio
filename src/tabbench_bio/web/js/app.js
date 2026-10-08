"use strict";

let DATA = null;
let EXCLUDED = new Set();
let SHOW_ALL_MODELS = false;
let RANK_REFERENCE_CELL = null;
let CHARTS_READY = false;

const SCRIPT_LOADS = new Map();
const TRAINING_OVERLAP_NOTE = "Part of the benchmark training data was used in the training process of this model.";
function hasTrainingOverlap(model) { return DATA.models[model.model_id || model.id].training_data_overlap === true; }
function modelLabel(model) { return model.display + (hasTrainingOverlap(model) ? " †" : ""); }
function overlapTooltip(model) { return hasTrainingOverlap(model) ? "<br>† " + TRAINING_OVERLAP_NOTE : ""; }

function loadExternalScript(path, globalName) {
  if (window[globalName]) return Promise.resolve();
  if (SCRIPT_LOADS.has(path)) return SCRIPT_LOADS.get(path);
  const promise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = path;
    script.async = true;
    script.addEventListener("load", () => {
      if (!window[globalName]) {
        reject(new Error(`${globalName} did not initialize`));
        return;
      }
      resolve();
    });
    script.addEventListener("error", () => reject(new Error(`Could not load ${path}`)));
    document.head.appendChild(script);
  });
  SCRIPT_LOADS.set(path, promise);
  return promise;
}

const ELO_METRICS = {
  f1_macro: "Macro-F1",
  matthews_corrcoef: "MCC",
  balanced_accuracy: "Balanced accuracy",
  roc_auc: "ROC-AUC",
};

const DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"];
const SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x"];
const PLOT_COLORS = {
  light: {
    "--ink": "#161a18", "--muted": "#616862", "--line": "#d9ded8",
    "--surface": "#ffffff", "--accent": "#176b52", "--violet": "#7357a8", "--amber": "#b46a14",
    "--elo-reference": "#111827",
  },
  dark: {
    "--ink": "#ffffff", "--muted": "#a7b0aa", "--line": "#314039",
    "--surface": "#161d19", "--accent": "#75c5a5", "--violet": "#b5a0dc", "--amber": "#e1aa61",
    "--elo-reference": "#ffffff",
  },
};

function byId(id) { return document.getElementById(id); }
function number(value) { return new Intl.NumberFormat("en-US").format(value); }
function percent(value) { return `${(100 * value).toFixed(1)}%`; }
function bytes(value) {
  const units = ["B", "KB", "MB", "GB"];
  let amount = Number(value), unit = 0;
  while (amount >= 1000 && unit < units.length - 1) { amount /= 1000; unit += 1; }
  return `${amount.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}
function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function capLabel(value) {
  if (value === null || String(value) === "full") return "Full";
  const parsed = Number(value);
  return parsed >= 1000 && parsed % 1000 === 0 ? `${parsed / 1000}k` : number(parsed);
}

function sampleLabel(value) {
  return value === null || String(value) === "full" ? "Full" : number(Number(value));
}

function budgetValue(value) {
  return value === null || value === undefined || String(value) === "null" || String(value) === "full"
    ? null
    : Number(value);
}

function sameBudget(left, right) {
  return budgetValue(left) === budgetValue(right);
}

function aboveRegularFeatureLimit(row, featureCap) {
  const limit = DATA.models[row.model_id].regular_max_features;
  const cap = budgetValue(featureCap);
  return limit !== null && limit !== undefined && cap !== null && cap > Number(limit);
}

function budgetSortValue(value) {
  const parsed = budgetValue(value);
  return parsed === null ? Number.POSITIVE_INFINITY : parsed;
}

function domainLabel(value) {
  return value === "all" ? "All" : value;
}

function domainDescription(value) {
  return value === "all" ? "All dataset modalities" : `${value} datasets`;
}

function cellShortLabel(cell) {
  const option = DATA.cell_options.find((entry) => entry.id === cell);
  if (!option) return cell;
  return `p=${capLabel(option.feature_cap)}, n=${sampleLabel(option.n_train)}`;
}

function css(name) {
  const palette = document.body.classList.contains("dark-theme") ? PLOT_COLORS.dark : PLOT_COLORS.light;
  if (!Object.hasOwn(palette, name)) throw new Error(`Unknown plot color: ${name}`);
  return palette[name];
}
function isMobileViewport() { return window.matchMedia("(max-width: 720px)").matches; }

function baseLayout(extra = {}) {
  return Object.assign({
    autosize: true,
    dragmode: false,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Atkinson Hyperlegible Next, ui-sans-serif, system-ui, sans-serif", size: 14, color: css("--ink") },
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--line"), font: { color: css("--ink") } },
    margin: { l: 80, r: 24, t: 24, b: 68 },
  }, extra);
}

function axes(axis = {}) {
  return Object.assign({
    fixedrange: true,
    gridcolor: css("--line"),
    linecolor: css("--line"),
    zerolinecolor: css("--line"),
    tickfont: { color: css("--muted"), size: 13 },
    titlefont: { color: css("--muted"), size: 13 },
    automargin: true,
  }, axis);
}

const PLOT_CONFIG = {
  responsive: true,
  displaylogo: false,
  scrollZoom: false,
  doubleClick: false,
  modeBarButtonsToRemove: [
    "toImage",
    "zoom2d", "pan2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
    "lasso2d", "select2d",
  ],
};

const COST_VIEWS = {
  cost: {
    column: "train_time_s",
    axis: "Median fit time per fold (s, log)",
    chart: "cost-chart",
  },
  "prediction-cost": {
    column: "inference_time_s",
    axis: "Median prediction time per fold (s, log)",
    chart: "prediction-cost-chart",
  },
};

const FIGURE_EXPORTS = [
  { id: "elo-chart", filename: "bradley-terry-elo", height: 640 },
  { id: "performance-chart", filename: "performance-across-sample-budgets", height: 640 },
  { id: "feature-chart", filename: "performance-across-feature-budgets", height: 640 },
  { id: "cost-chart", filename: "performance-vs-fitting-cost", height: 640 },
  { id: "prediction-cost-chart", filename: "performance-vs-prediction-cost", height: 640 },
  { id: "rank-chart", filename: "rank-correlation-across-cells", height: 600 },
  { id: "composition-chart", filename: "datasets-by-modality", height: 600 },
];

async function figureBlobs(plot, height) {
  if (plot.layout.meta?.cost) {
    const { rows, metric, timing } = plot.layout.meta.cost;
    const spec = costPlotSpec(rows, metric, timing, false, 1100);
    const visibility = new Map(plot.data.map((trace) => [trace.name, trace.visible]));
    spec.traces.forEach((trace) => { trace.visible = visibility.get(trace.name); });
    const hidden = new Set(spec.traces.filter((trace) => trace.visible === "legendonly" || trace.visible === false).flatMap((trace) => trace.ids || []));
    spec.layout.annotations.forEach((label) => { label.visible = !hidden.has(label.name); });
    plot = { data: spec.traces, layout: spec.layout };
    height = spec.layout.height;
  }
  const svg = await Plotly.toImage(plot, {
    format: "svg",
    width: 1100,
    height,
    scale: 1,
    imageDataOnly: true,
  });
  const svgBlob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
  const image = new Image();
  const canvas = document.createElement("canvas");
  try {
    // A data URL also works in Safari, where SVG blob URLs can taint a canvas.
    image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
    await image.decode();
    canvas.width = 2200;
    canvas.height = height * 2;
    canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
    const png = await new Promise((resolve, reject) => {
      canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("PNG encoding failed")), "image/png");
    });
    return { png, svg: svgBlob };
  } finally {
    canvas.width = canvas.height = 0;
    image.removeAttribute("src");
  }
}

async function addFigure(folder, filename, plot, height) {
  const blobs = await figureBlobs(plot, height);
  folder.file(`${filename}.png`, blobs.png);
  folder.file(`${filename}.svg`, blobs.svg);
}

async function runFigureJobs(jobs, onProgress) {
  let next = 0;
  let completed = 0;
  let failed = false;
  async function run() {
    while (!failed && next < jobs.length) {
      const job = jobs[next++];
      try {
        await job();
        onProgress(++completed);
        // Let input and progress updates paint between figures.
        await new Promise((resolve) => setTimeout(resolve, 0));
      } catch (error) {
        failed = true;
        throw error;
      }
    }
  }
  // PNG encoding is latency-bound rather than CPU-bound, so several figures stay in
  // flight; the ceiling bounds live canvases and Plotly clones. Drain on failure.
  const lanes = Math.min(jobs.length, Math.max(4, Math.min(8, navigator.hardwareConcurrency || 4)));
  const results = await Promise.allSettled(Array.from({ length: lanes }, run));
  const failure = results.find((result) => result.status === "rejected");
  if (failure) throw failure.reason;
}

function filenameSlug(value) {
  return String(value).toLowerCase().replaceAll(/[^a-z0-9]+/g, "-").replaceAll(/^-|-$/g, "");
}

function eloGridFigures() {
  const groups = new Map();
  DATA.domain_elo
    .filter((row) => !EXCLUDED.has(row.model_id))
    .forEach((row) => {
      const key = `${row.metric}|${row.cell}|${row.domain}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
    });
  const metricOrder = new Map(DATA.elo_metrics.map((value, index) => [value, index]));
  const cellOrder = new Map(DATA.cell_options.map((value, index) => [value.id, index]));
  const domainOrder = new Map(DATA.domains.map((value, index) => [value, index]));
  return [...groups.entries()]
    .map(([key, rows]) => {
      const [metric, cell, domain] = key.split("|");
      return {
        metric,
        cell,
        domain,
        rows: rows.sort((a, b) => a.Elo - b.Elo),
        featureCap: DATA.cell_options.find((option) => option.id === cell).feature_cap,
        filename: `elo_${filenameSlug(metric)}_${filenameSlug(cell)}_${filenameSlug(domain)}`,
      };
    })
    .sort((a, b) => metricOrder.get(a.metric) - metricOrder.get(b.metric)
      || cellOrder.get(a.cell) - cellOrder.get(b.cell)
      || domainOrder.get(a.domain) - domainOrder.get(b.domain));
}

async function addEloGridFigure(folder, figure) {
  const spec = eloPlotSpec(figure.rows, false, figure.featureCap);
  await addFigure(folder, `grid/elo-cells/${figure.filename}`, { data: spec.traces, layout: spec.layout }, spec.layout.height);
}

function costGridFigures() {
  const groups = new Map();
  DATA.cost_grid
    .filter((row) => !EXCLUDED.has(row.model_id) && (SHOW_ALL_MODELS || row.model_id !== "AUTOGLUON"))
    .forEach((row) => {
      const key = `${row.cell}|${row.domain}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
    });
  const metricOrder = new Map(DATA.elo_metrics.map((value, index) => [value, index]));
  const cellOrder = new Map(DATA.cell_options.map((value, index) => [value.id, index]));
  const domainOrder = new Map(DATA.domains.map((value, index) => [value, index]));
  return [...groups.entries()].flatMap(([key, rows]) => {
    const [cell, domain] = key.split("|");
    return DATA.elo_metrics.map((metric) => ({
      metric,
      cell,
      domain,
      rows,
      filename: `pareto_${filenameSlug(metric)}_${filenameSlug(cell)}_${filenameSlug(domain)}`,
    }));
  }).sort((a, b) => metricOrder.get(a.metric) - metricOrder.get(b.metric)
    || cellOrder.get(a.cell) - cellOrder.get(b.cell)
    || domainOrder.get(a.domain) - domainOrder.get(b.domain));
}

async function addCostGridFigure(folder, figure) {
  const spec = costPlotSpec(figure.rows, figure.metric, COST_VIEWS.cost, false);
  await addFigure(folder, `grid/pareto-cells/${figure.filename}`, { data: spec.traces, layout: spec.layout }, spec.layout.height);
}

async function downloadAllFigures(button) {
  const originalLabel = button.textContent;
  button.disabled = true;
  try {
    button.textContent = "Loading export tools…";
    await loadExternalScript("assets/jszip.min.js", "JSZip");
    const zip = new JSZip();
    const folder = zip.folder("tabbench-bio-figures");
    const eloFigures = eloGridFigures();
    const costFigures = costGridFigures();
    const total = FIGURE_EXPORTS.length + eloFigures.length + costFigures.length;
    // Copy the six selected views before asynchronous export work starts.
    const jobs = FIGURE_EXPORTS.map((figure) => {
      const plot = byId(figure.id);
      const snapshot = { data: structuredClone(plot.data), layout: structuredClone(plot.layout) };
      return () => addFigure(folder, figure.filename, snapshot, figure.height);
    });
    jobs.push(...eloFigures.map((figure) => () => addEloGridFigure(folder, figure)));
    jobs.push(...costFigures.map((figure) => () => addCostGridFigure(folder, figure)));
    await runFigureJobs(jobs, (completed) => {
      button.textContent = `Preparing figures… ${completed}/${total}`;
    });
    button.textContent = "Packaging figures…";
    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "tabbench-bio-figures.zip";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } finally {
    button.disabled = false;
    button.textContent = originalLabel;
  }
}

function setOptions(select, options, current, labeler = (value) => value) {
  select.innerHTML = "";
  options.forEach((value) => {
    const option = document.createElement("option");
    option.value = String(value);
    option.textContent = labeler(value);
    select.appendChild(option);
  });
  if (current !== undefined && options.map(String).includes(String(current))) select.value = String(current);
}

function renderReferencePodium() {
  const leaders = DATA.reference
    .filter((row) => !EXCLUDED.has(row.model_id) && (SHOW_ALL_MODELS || row.model_id !== "AUTOGLUON") && !hasTrainingOverlap(row))
    .sort((a, b) => b.Elo - a.Elo)
    .slice(0, 3);
  const setting = `Macro-F1 Elo · ${leaders[0].cell_label} · ${number(leaders[0].n_targets)} targets`;
  byId("podium-list").innerHTML = leaders.map((row, index) => `
    <li class="podium-place podium-place-${index + 1}">
      <span class="podium-contender">
        <span class="podium-model"><strong>${escapeHtml(row.display)}</strong><span>${escapeHtml(row.category)}</span></span>
        <span class="podium-score"><strong>${number(row.Elo)}</strong><span>CI ${number(row.Elo_lo)}–${number(row.Elo_hi)}</span></span>
      </span>
      <span class="podium-step" aria-label="Rank ${index + 1}">${index + 1}</span>
    </li>`).join("");
  const intervalsOverlap = leaders.slice(1).some((row) => row.Elo_hi >= leaders[0].Elo_lo);
  byId("podium-note").textContent = intervalsOverlap
    ? `Top-rated at the reference setting (${setting}); 95% intervals overlap, so the ordering is descriptive.`
    : `Top-rated at the reference setting (${setting}); rankings can change with the data budget.`;
}

function initializeMeta() {
  const { meta, progress } = DATA;
  [byId("github-link"), byId("nav-github")].forEach((link) => {
    link.href = meta.github_url;
    link.target = "_blank";
    link.rel = "noreferrer";
  });
  const requestLink = byId("request-link");
  requestLink.target = "_blank";
  requestLink.rel = "noreferrer";
  const paperLink = byId("paper-link");
  paperLink.href = meta.paper_url;
  if (meta.paper_url.startsWith("http")) {
    paperLink.target = "_blank";
    paperLink.rel = "noreferrer";
  }
  byId("footer-affiliation").firstChild.textContent = `${meta.affiliation} `;
  renderReferencePodium();

  byId("snapshot-date").textContent = new Date(meta.snapshot_utc).toLocaleDateString("en-GB", {
    day: "numeric", month: "short", year: "numeric", timeZone: "UTC",
  });
  byId("progress-percent").textContent = percent(progress.fraction);
  const progressTrack = byId("progress-track");
  progressTrack.setAttribute("aria-valuenow", (100 * progress.fraction).toFixed(1));
  progressTrack.setAttribute("aria-valuetext", `${percent(progress.fraction)} of planned units recorded`);
  byId("pass-count").textContent = number(progress.status.pass);
  byId("skip-count").textContent = number(progress.status.skip);
  byId("fail-count").textContent = number(progress.status.fail);
  byId("progress-pass").style.width = percent(progress.status.pass / progress.expected);
  byId("progress-skip").style.width = percent(progress.status.skip / progress.expected);
  byId("progress-fail").style.width = percent(progress.status.fail / progress.expected);
  byId("cell-count").textContent = `${DATA.cell_options.length} cells are shown from recorded metrics; ${progress.cells_status_complete} are status-complete in this snapshot.`;
  const recordedFoldCounts = DATA.datasets.flatMap((dataset) =>
    dataset.performance.models.map((model) => model.folds)
  );
  const cvFolds = Math.max(...recordedFoldCounts);
  const evaluationPointsPerModel = meta.evaluation_points_per_model
    ?? cvFolds * progress.cells.length * DATA.datasets.length;
  const configuredModelCount = meta.configured_model_count ?? Math.round(
    Math.max(...progress.cells.map((cell) => cell.expected)) / (cvFolds * DATA.datasets.length)
  );
  byId("dataset-count").textContent = number(DATA.datasets.length);
  byId("model-count").textContent = number(configuredModelCount);
  byId("evaluation-points-per-model").textContent = number(evaluationPointsPerModel);
}

function modelCardScale(delta) {
  // A fixed scale keeps colours comparable across models, caps and analysis views.
  return Math.max(-1, Math.min(1, Math.asinh(delta / 200) / Math.asinh(1000 / 200)));
}

function modelCardColor(delta) {
  const amount = Math.abs(modelCardScale(delta)) * 100;
  const endpoint = delta < 0 ? "--model-card-negative" : "--model-card-positive";
  return `color-mix(in oklab, var(${endpoint}) ${amount}%, var(--surface-strong))`;
}

function renderModelTuning(model) {
  const notice = byId("model-card-disclaimer");
  const panel = byId("model-card-tuning");
  panel.hidden = !model.tuned_from || !model.tuning?.grid;
  panel.innerHTML = "";
  if (!model.tuned_from) {
    notice.innerHTML = "<strong>Default parameters.</strong> This model uses its prespecified benchmark-default configuration without per-dataset tuning. This card describes that configuration, not the best achievable tuned model.";
    return;
  }
  const tuning = model.tuning;
  if (!tuning?.grid) {
    notice.innerHTML = "<strong>Tuned configuration.</strong> The search grid is not available in this result export.";
    return;
  }
  const budget = tuning.budget_seconds
    ? ` under a ${tuning.budget_seconds / 3600 === 1 ? "one-hour" : `${tuning.budget_seconds / 60}-minute`} budget`
    : "";
  notice.innerHTML = `<strong>Tuned configuration.</strong> A small predefined grid${escapeHtml(budget)}. Candidates are selected on a ${100 * tuning.validation_fraction}% training-only holdout; the winner is refitted on the full training fold. Up to ${100 * tuning.search_fraction}% of the budget is used for search, with the remainder reserved for refitting.`;
  const excludesDefaults = tuning.excluded_candidates?.some((candidate) => Object.keys(candidate).length === 0);
  if (excludesDefaults) notice.innerHTML += " Outcomes that selected benchmark defaults have been removed; these results are a filtered subset of the original run, not a new tuning run.";
  const taskGrids = Object.hasOwn(tuning.grid, "classification")
    ? Object.entries(tuning.grid)
    : [["Search grid", tuning.grid]];
  const count = (block) => Object.values(block).reduce((total, values) => total * values.length, 1);
  const tables = taskGrids.map(([label, grid]) => {
    const blocks = (Array.isArray(grid) ? grid : [grid])
      .filter((block) => !excludesDefaults || Object.keys(block).length > 0);
    const total = blocks.reduce((sum, block) => sum + count(block), 0);
    const rows = blocks.map((block) => {
      const parameters = Object.entries(block).map(([key, values]) =>
        `<span class="model-grid-param"><code>${escapeHtml(key)}</code>: ${values.map((value) => `<code>${escapeHtml(JSON.stringify(value))}</code>`).join(", ")}</span>`
      ).join("") || "Benchmark defaults (no overrides)";
      return `<tr><td>${count(block)}</td><td>${parameters}</td></tr>`;
    }).join("");
    return `<table><caption>${escapeHtml(label)} · ${total} ${excludesDefaults ? "retained " : ""}candidates</caption><thead><tr><th scope="col">Candidates</th><th scope="col">Parameter values</th></tr></thead><tbody>${rows}</tbody></table>`;
  }).join("");
  panel.innerHTML = `<summary>Grid parameters</summary><p>Each row adds a separate set of candidates. Values within a row are crossed; unspecified parameters keep their benchmark defaults. The budget may end before every candidate is tried.</p>${tables}`;
}

function renderModelCard() {
  const modelId = byId("model-card-model").value;
  const comparatorId = byId("model-card-comparator").value;
  const comparator = DATA.models[comparatorId];
  const featureCap = budgetValue(byId("model-card-cap").value);
  const model = DATA.models[modelId];
  const domains = DATA.domains.filter((domain) => domain !== "all");
  const cells = DATA.cell_options
    .filter((cell) => sameBudget(cell.feature_cap, featureCap))
    .sort((left, right) => budgetSortValue(left.n_train) - budgetSortValue(right.n_train));
  const rows = DATA.domain_elo.filter((row) =>
    row.model_id === modelId && row.metric === "f1_macro" && domains.includes(row.domain)
  );
  const byCell = new Map(rows.map((row) => [`${row.domain}|${row.cell}`, row]));
  const comparatorByCell = new Map(DATA.domain_elo
    .filter((row) => row.model_id === comparatorId && row.metric === "f1_macro")
    .map((row) => [`${row.domain}|${row.cell}`, row]));
  const comparatorCoverage = new Map(DATA.model_card_coverage
    .filter((row) => row.model_id === comparatorId)
    .map((row) => [`${row.domain}|${row.cell}`, row]));
  const coverageByCell = new Map(DATA.model_card_coverage
    .filter((row) => row.model_id === modelId && domains.includes(row.domain))
    .map((row) => [`${row.domain}|${row.cell}`, row]));

  renderModelTuning(model);
  byId("model-card-name").textContent = modelLabel(model);
  byId("model-card-caption").textContent = `Macro-F1 Elo relative to ${comparator.display}. Columns are training-sample caps; rows are dataset modalities. Small labels show the number of binding targets.`;
  byId("model-card-scale-label").textContent = `Elo difference vs. ${comparator.display}`;
  byId("model-card-family").textContent = model.category;
  byId("model-card-limit").textContent = model.regular_max_features === null || model.regular_max_features === undefined
    ? "No declared regular feature limit"
    : `Regular feature limit: ${capLabel(model.regular_max_features)}`;
  const overlapWarning = byId("model-card-overlap-warning");
  overlapWarning.hidden = !hasTrainingOverlap(model);
  overlapWarning.textContent = `† ${model.display}: ${TRAINING_OVERLAP_NOTE}`;
  byId("model-card-name").title = hasTrainingOverlap(model) ? TRAINING_OVERLAP_NOTE : "";

  const grid = byId("model-card-grid");
  grid.style.setProperty("--model-card-sample-count", cells.length);
  const header = `<div class="model-card-corner" role="columnheader">Modality</div>${cells.map((cell) =>
    `<div class="model-card-column" role="columnheader">n=${escapeHtml(sampleLabel(cell.n_train))}</div>`
  ).join("")}`;
  const observations = [];
  const body = domains.map((domain) => {
    const values = cells.map((cell) => {
      const row = byCell.get(`${domain}|${cell.id}`);
      const coverage = coverageByCell.get(`${domain}|${cell.id}`);
      if (coverage && coverage.successful_fits === 0 && coverage.failed_fits > 0) {
        const detail = `${domain}, ${cell.label}: no data. Every recorded fit failed, for example because of an out-of-memory condition or another training error. Failures are scored at chance in the primary analysis.`;
        return `<div class="model-card-cell model-card-cell-missing" role="cell" aria-label="${escapeHtml(detail)}" title="${escapeHtml(detail)}"><strong>No data</strong><span>Fit error</span></div>`;
      }
      if (!row) {
        return `<div class="model-card-cell model-card-cell-missing" role="cell" aria-label="${escapeHtml(domain)}, ${escapeHtml(cell.label)}: no nominal result"><strong>—</strong><span>No result</span></div>`;
      }
      const comparison = comparatorByCell.get(`${domain}|${cell.id}`);
      const comparisonCoverage = comparatorCoverage.get(`${domain}|${cell.id}`);
      const comparisonFailed = comparisonCoverage && comparisonCoverage.successful_fits === 0 && comparisonCoverage.failed_fits > 0;
      if (!comparison || comparisonFailed) {
        const reason = comparisonFailed ? "every recorded fit failed" : "no nominal result";
        const detail = `${domain}, ${cell.label}: comparison unavailable for ${comparator.display}: ${reason}.`;
        return `<div class="model-card-cell model-card-cell-missing" role="cell" aria-label="${escapeHtml(detail)}" title="${escapeHtml(detail)}"><strong>—</strong><span>Comparison unavailable</span></div>`;
      }
      const delta = row.Elo - comparison.Elo;
      observations.push({ domain, cell, row, delta });
      const signed = delta > 0 ? `+${number(Math.round(delta))}` : number(Math.round(delta));
      const failureDetail = coverage && coverage.failed_fits > 0
        ? `; ${number(coverage.failed_fits)} of ${number(coverage.successful_fits + coverage.failed_fits)} fits failed and were scored at chance`
        : "";
      const comparisonFailureDetail = comparisonCoverage && comparisonCoverage.failed_fits > 0
        ? `; ${comparator.display}: ${number(comparisonCoverage.failed_fits)} of ${number(comparisonCoverage.successful_fits + comparisonCoverage.failed_fits)} fits failed and were scored at chance`
        : "";
      const intervals = modelId === comparatorId ? "same model; difference is zero by construction"
        : comparatorId === "RF"
          ? `difference 95% interval ${number(row.Elo_lo - comparison.Elo)} to ${number(row.Elo_hi - comparison.Elo)}`
          : `${model.display} Elo ${number(row.Elo)} (95% interval ${number(row.Elo_lo)} to ${number(row.Elo_hi)}); ${comparator.display} Elo ${number(comparison.Elo)} (95% interval ${number(comparison.Elo_lo)} to ${number(comparison.Elo_hi)}); intervals describe individual ratings, not their difference`;
      const detail = `${domain}, ${cell.label}: ${signed} Elo relative to ${comparator.display}; ${intervals}; ${number(row.n_targets)} binding targets${failureDetail}${comparisonFailureDetail}`;
      const textColor = modelCardScale(delta) < -0.65 ? "#ffffff" : "var(--ink)";
      return `<div class="model-card-cell" style="background:${modelCardColor(delta)};color:${textColor}" role="cell" aria-label="${escapeHtml(detail)}" title="${escapeHtml(detail)}"><strong>${signed}</strong><span>${number(row.n_targets)} target${row.n_targets === 1 ? "" : "s"}</span></div>`;
    }).join("");
    return `<div class="model-card-row" role="rowheader">${escapeHtml(domain)}</div>${values}`;
  }).join("");
  grid.innerHTML = header + body;

  if (!observations.length) {
    byId("model-card-summary").textContent = `No comparable Macro-F1 rankings are available for ${model.display} and ${comparator.display} at this feature cap.`;
    return;
  }
  if (modelId === comparatorId) {
    byId("model-card-summary").textContent = `${model.display} is compared with itself, so every available cell is zero by construction.`;
    return;
  }
  byId("model-card-summary").textContent = `Relative to ${comparator.display} at ${capLabel(featureCap)} features.`;
}

function initializeModelCard() {
  const modelSelect = byId("model-card-model");
  const capSelect = byId("model-card-cap");
  const comparatorSelect = byId("model-card-comparator");
  const modelIds = Object.keys(DATA.models)
    .filter((modelId) => !EXCLUDED.has(modelId) && (SHOW_ALL_MODELS || modelId !== "AUTOGLUON"))
    .sort((left, right) => DATA.models[left].display.localeCompare(DATA.models[right].display));
  const defaultModel = DATA.reference
    .filter((row) => modelIds.includes(row.model_id))
    .sort((left, right) => right.Elo - left.Elo)[0].model_id;
  const caps = [...new Set(DATA.cell_options.map((cell) => cell.feature_cap))]
    .sort((left, right) => budgetSortValue(left) - budgetSortValue(right));
  const reference = DATA.cell_options.find((cell) => cell.id === DATA.meta.reference_cell);
  setOptions(modelSelect, modelIds, modelIds.includes(modelSelect.value) ? modelSelect.value : defaultModel, (modelId) => DATA.models[modelId].display);
  setOptions(comparatorSelect, modelIds, comparatorSelect.value || "RF", (modelId) => DATA.models[modelId].display);
  setOptions(capSelect, caps, reference.feature_cap, capLabel);
  const ticks = [-1000, -300, 0, 300, 1000];
  byId("model-card-scale-ticks").innerHTML = ticks.map((value) => {
    const label = value === -1000 ? "≤−1,000" : value === 1000 ? "≥+1,000" : value > 0 ? `+${value}` : String(value).replace("-", "−");
    return `<span style="left:${50 + 50 * modelCardScale(value)}%">${label}</span>`;
  }).join("");
  comparatorSelect.onchange = renderModelCard;
  modelSelect.onchange = renderModelCard;
  capSelect.onchange = renderModelCard;
  renderModelCard();
}

function renderFamilyLegend(rows, hasTuned = false) {
  const categories = [...new Set(rows.map((row) => row.category))];
  const entries = categories.map((category) => {
    const model = rows.find((row) => row.category === category);
    return `<span><i style="background:${escapeHtml(model.color)}"></i>${escapeHtml(category)}</span>`;
  });
  if (rows.length) entries.push('<span><i class="legend-hatch" aria-hidden="true"></i>Diagonal bars: over regular feature limit</span>');
  if (hasTuned) entries.push(
    '<span><i class="legend-tuning-change" aria-hidden="true"></i>Tuning change</span>',
    '<span><i class="legend-tuned-elo" aria-hidden="true"></i>Tuned Elo</span>',
  );
  byId("family-legend").innerHTML = entries.join("");
  const flagged = rows.filter(hasTrainingOverlap);
  const warning = byId("elo-overlap-warning");
  warning.hidden = flagged.length === 0;
  warning.textContent = `† ${flagged.map((row) => row.display).join(", ")}: Part of the benchmark training data was used in the training process of ${flagged.length === 1 ? "this model" : "these models"}.`;
}

function initializeElo() {
  const metricSelect = byId("elo-metric");
  const domainSelect = byId("elo-domain");
  const capSelect = byId("elo-cap");
  const sampleSelect = byId("elo-samples");
  const caps = [...new Set(DATA.cell_options.map((cell) => cell.feature_cap))].sort((a, b) => Number(a) - Number(b));
  const reference = DATA.cell_options.find((cell) => cell.id === DATA.meta.reference_cell) || DATA.cell_options[0];
  setOptions(metricSelect, DATA.elo_metrics, "f1_macro", (metric) => ELO_METRICS[metric]);
  setOptions(domainSelect, DATA.domains, "all", domainLabel);
  setOptions(capSelect, caps, reference.feature_cap, capLabel);

  function updateSamples(preferred) {
    const cap = budgetValue(capSelect.value);
    const samples = DATA.cell_options
      .filter((cell) => sameBudget(cell.feature_cap, cap))
      .map((cell) => cell.n_train)
      .sort((a, b) => Number(a) - Number(b));
    setOptions(sampleSelect, samples, preferred, sampleLabel);
  }

  metricSelect.addEventListener("change", renderElo);
  domainSelect.addEventListener("change", renderElo);
  capSelect.addEventListener("change", () => { updateSamples(sampleSelect.value); renderElo(); });
  sampleSelect.addEventListener("change", renderElo);
  updateSamples(reference.n_train);
  renderElo();
}

function nonnegativeEloInterval(row) {
  if (row.Elo_hi < 0) return null;
  const center = Math.max(0, row.Elo);
  return { center, plus: Math.max(0, row.Elo_hi) - center, minus: center - Math.max(0, row.Elo_lo) };
}

function eloPlotSpec(rows, mobile, featureCap) {
  const byModel = new Map(rows.map((row) => [row.model_id, row]));
  const paired = rows.filter((row) => byModel.has(DATA.models[row.model_id].tuned_from));
  const pairedIds = new Set(paired.flatMap((row) => [row.model_id, DATA.models[row.model_id].tuned_from]));
  const visible = rows.filter((row) => SHOW_ALL_MODELS || row.Elo >= 0 || pairedIds.has(row.model_id));
  const autogluon = visible.find((row) => row.model_id === "AUTOGLUON");
  const tunedByParent = new Map(paired.map((row) => [DATA.models[row.model_id].tuned_from, row]));
  const rankingElo = (row) => Math.max(row.Elo, (tunedByParent.get(row.model_id) || row).Elo);
  const peers = visible.filter((row) => row.model_id !== "AUTOGLUON" && !paired.includes(row))
    .sort((a, b) => rankingElo(a) - rankingElo(b));
  const aboveLimit = peers.map((row) => aboveRegularFeatureLimit(row, featureCap));
  const trace = {
    type: "bar",
    orientation: "h",
    width: 0.72,
    showlegend: false,
    x: peers.map((row) => row.Elo),
    y: peers.map(modelLabel),
    marker: {
      color: peers.map((row) => row.color),
      line: { color: css("--surface"), width: 0.5 },
      pattern: { shape: aboveLimit.map((value) => value ? "/" : ""), size: 12, solidity: 0.5 },
    },
    error_x: {
      type: "data",
      symmetric: false,
      array: peers.map((row) => tunedByParent.has(row.model_id) || row.Elo < 0 ? null : nonnegativeEloInterval(row)?.plus),
      arrayminus: peers.map((row) => tunedByParent.has(row.model_id) || row.Elo < 0 ? null : nonnegativeEloInterval(row)?.minus),
      color: css("--muted"), thickness: 1, width: 3,
    },
    customdata: peers.map((row, index) => [
      row.Elo_lo,
      row.Elo_hi,
      row.category,
      overlapTooltip(row),
      rankingElo(row),
      (tunedByParent.get(row.model_id) || row).Elo,
    ]),
    hovertemplate: peers.map((row) => tunedByParent.has(row.model_id)
      ? "<b>%{y}</b><br>Ranking Elo (max) %{customdata[4]:.0f}<br>Untuned Elo %{x:.0f}<br>Tuned Elo %{customdata[5]:.0f}<br>%{customdata[2]}%{customdata[3]}<extra></extra>"
      : "<b>%{y}</b><br>Elo %{x:.0f}<br>95% interval [%{customdata[0]:.0f}, %{customdata[1]:.0f}]<br>%{customdata[2]}%{customdata[3]}<extra></extra>"),
  };
  const traces = [trace];
  if (paired.length) {
    const parents = paired.map((row) => byModel.get(DATA.models[row.model_id].tuned_from));
    const labels = parents.map(modelLabel);
    const details = paired.map((row, i) => [
      modelLabel(row), parents[i].Elo, row.Elo, row.Elo - parents[i].Elo, row.Elo_lo, row.Elo_hi, overlapTooltip(row),
    ]);
    const hover = "<b>%{customdata[0]}</b><br>Untuned Elo %{customdata[1]:.0f}<br>Tuned Elo %{customdata[2]:.0f}<br>Tuned 95% interval [%{customdata[4]:.0f}, %{customdata[5]:.0f}]%{customdata[6]}<extra></extra>";
    // Explicit bases preserve signed changes: the segment ends at tuned Elo, never their sum.
    traces.push({
      type: "bar", orientation: "h", name: "Tuning change", width: 0.72,
      base: parents.map((row) => row.Elo),
      x: paired.map((row, i) => row.Elo - parents[i].Elo), y: labels,
      marker: {
        color: paired.map((row, i) => row.Elo < parents[i].Elo ? css("--amber") : row.color),
        line: { color: css("--ink"), width: 0.7 },
        pattern: { shape: ".", size: 5, solidity: 0.25, bgcolor: paired.map(() => css("--surface")) },
      },
      customdata: details, hoverinfo: "skip", showlegend: false,
    }, {
      type: "scatter", mode: "markers", name: "Tuned Elo", showlegend: false,
      x: paired.map((row) => row.Elo), y: labels,
      marker: { symbol: "diamond", size: 7, color: css("--ink") },
      error_x: {
        type: "data", symmetric: false,
        array: paired.map((row) => row.Elo < 0 ? null : nonnegativeEloInterval(row)?.plus),
        arrayminus: paired.map((row) => row.Elo < 0 ? null : nonnegativeEloInterval(row)?.minus),
        color: css("--ink"), thickness: 1, width: 3,
      },
      customdata: details, hovertemplate: hover,
    });
  }
  const crossing = [...peers.filter((row) => !tunedByParent.has(row.model_id)), ...paired]
    .filter((row) => row.Elo < 0 && nonnegativeEloInterval(row));
  if (crossing.length) traces.push({
    type: "scatter", mode: "markers", showlegend: false, hoverinfo: "skip",
    x: crossing.map(() => 0),
    y: crossing.map((row) => paired.includes(row) ? modelLabel(byModel.get(DATA.models[row.model_id].tuned_from)) : modelLabel(row)),
    marker: { size: 0 },
    error_x: {
      type: "data", symmetric: false, array: crossing.map((row) => nonnegativeEloInterval(row).plus),
      arrayminus: crossing.map(() => 0), color: css("--ink"), thickness: 1, width: 3,
    },
  });
  const referenceValues = autogluon ? [autogluon.Elo, autogluon.Elo_hi] : [];
  const xMax = Math.ceil(Math.max(1000, ...visible.map((row) => tunedByParent.has(row.model_id) ? row.Elo : row.Elo_hi), ...referenceValues) / 100) * 100;
  const xMin = Math.floor(Math.min(0, ...visible.map((row) => row.Elo < 0 || tunedByParent.has(row.model_id) ? row.Elo : Math.max(0, row.Elo_lo))) / 100) * 100;
  const shapes = [{
    type: "line", x0: 1000, x1: 1000, y0: -0.5, y1: peers.length - 0.5,
    line: { color: css("--ink"), dash: "dot", width: 1.2 },
  }];
  const annotations = [];
  if (autogluon) {
    shapes.push({
      type: "line", x0: autogluon.Elo, x1: autogluon.Elo, y0: -0.5, y1: peers.length - 0.5,
      line: { color: css("--elo-reference"), dash: "solid", width: 2.2 },
    });
    annotations.push({
      x: autogluon.Elo, y: 1, xref: "x", yref: "paper",
      xanchor: "center", yanchor: "bottom",
      text: `AutoGluon 1 h · ${Math.round(autogluon.Elo)}`, showarrow: false,
      font: { color: css("--elo-reference"), size: mobile ? 10 : 11 },
      bgcolor: css("--surface"), borderpad: 2,
    });
  }
  const layout = baseLayout({
    height: Math.max(mobile ? 650 : 570, 31 * peers.length + (mobile ? 155 : 115)),
    margin: { l: mobile ? 112 : 150, r: mobile ? 18 : 30, t: autogluon ? 38 : 18, b: mobile ? 94 : 62 },
    bargap: 0.24,
    barmode: "overlay",
    showlegend: false,
    xaxis: axes({ title: "Bradley–Terry Elo (untuned Random Forest = 1,000)", range: [xMin, xMax] }),
    yaxis: axes({
      showgrid: false, categoryorder: "array", categoryarray: peers.map(modelLabel),
      tickmode: "array", tickvals: peers.map(modelLabel),
      ticktext: peers.map(modelLabel),
      tickfont: { color: css("--ink"), size: 13 },
    }),
    shapes,
    annotations,
  });
  return {
    traces,
    layout,
    peers,
    autogluon,
    paired,
    hiddenNegativeCount: rows.length - visible.length,
  };
}

function renderElo() {
  const mobile = isMobileViewport();
  const metric = byId("elo-metric").value;
  byId("elo-metric-note").hidden = metric === "f1_macro";
  const domain = byId("elo-domain").value;
  const cap = budgetValue(byId("elo-cap").value);
  const samples = budgetValue(byId("elo-samples").value);
  const option = DATA.cell_options.find((cell) => sameBudget(cell.feature_cap, cap) && sameBudget(cell.n_train, samples));
  if (!option) return;
  const rows = DATA.domain_elo
    .filter((row) => row.cell === option.id && row.domain === domain && row.metric === metric && !EXCLUDED.has(row.model_id))
    .sort((a, b) => a.Elo - b.Elo);
  const chart = byId("elo-chart");
  if (!rows.length) {
    renderFamilyLegend([]);
    Plotly.purge(chart);
    chart.innerHTML = `<p style="margin:0;padding:96px 24px;color:var(--muted);text-align:center">No ${escapeHtml(domainDescription(domain).toLowerCase())} results are available for ${escapeHtml(option.label)}.</p>`;
    byId("elo-note").textContent = `${ELO_METRICS[metric]} · ${option.label} · ${domainDescription(domain)} · no ranking available.`;
    return;
  }
  const spec = eloPlotSpec(rows, mobile, option.feature_cap);
  renderFamilyLegend([...spec.peers, ...spec.paired], spec.paired.length > 0);
  if (Array.isArray(chart.data)) {
    Plotly.react(chart, spec.traces, spec.layout, PLOT_CONFIG);
  } else {
    chart.replaceChildren();
    Plotly.newPlot(chart, spec.traces, spec.layout, PLOT_CONFIG);
  }
  const referenceNote = spec.autogluon ? " · solid line: AutoGluon 1 h reference" : "";
  const hiddenNote = spec.hiddenNegativeCount
    ? ` · ${spec.hiddenNegativeCount} negative-Elo model${spec.hiddenNegativeCount === 1 ? "" : "s"} hidden`
    : "";
  const tuningNote = spec.paired.length
    ? " Solid bars show untuned Elo; dotted segments show the tuning change, ending at the tuned diamond. Decreases point left and are amber. Rows are ordered by the higher of untuned and tuned Elo; both original ratings remain unchanged. For paired models, only the tuned rating has an interval. Tuning uses a small predefined grid under a one-hour budget."
    : "";
  byId("elo-note").textContent = `${ELO_METRICS[metric]} · ${option.label} · ${domainDescription(domain)} · ${spec.peers.length ? spec.peers[0].n_targets : 0} binding targets · untuned Random Forest is anchored at 1,000${hiddenNote} · error bars are target-bootstrap 95% intervals${referenceNote}. Displayed intervals are clipped at zero; hover shows the original bounds.${tuningNote}`;
}

const BUDGET_VIEWS = {
  perf: {
    chart: "performance-chart",
    note: "performance-note",
    controlId: "perf-cap",
    controlOf: (cell) => cell.feature_cap,
    controlDefault: 10000,
    controlLabel: capLabel,
    axisOf: (cell) => cell.n_train,
    axisLabel: sampleLabel,
    axisTitle: "Training samples (log scale)",
    axisType: "log",
    includes: (cell) => budgetValue(cell.n_train) !== null,
    hover: "Training samples",
    describe: (control) => `${capLabel(control)} features`,
    budgetNote: "training-sample budget",
    leaders: "reference",
    referenceBudget: 100,
  },
  feat: {
    chart: "feature-chart",
    note: "feature-note",
    controlId: "feat-samples",
    controlOf: (cell) => cell.n_train,
    controlDefault: 100,
    controlLabel: sampleLabel,
    axisOf: (cell) => cell.feature_cap,
    axisLabel: capLabel,
    axisTitle: "Feature budget",
    axisType: "category",
    includes: () => true,
    hover: "Feature budget",
    describe: (control) => `${sampleLabel(control)} training samples`,
    budgetNote: "feature budget",
    leaders: "mean",
  },
};

function budgetRank(value) {
  return value === null ? Infinity : value;
}

function initializeBudget(prefix) {
  const view = BUDGET_VIEWS[prefix];
  const rows = DATA.domain_elo.filter((row) => row.metric === "f1_macro" && !EXCLUDED.has(row.model_id) && (SHOW_ALL_MODELS || row.model_id !== "AUTOGLUON"));
  setOptions(byId(`${prefix}-domain`), DATA.domains, "all", domainLabel);
  const controls = [...new Set(DATA.cell_options.map((cell) => budgetValue(view.controlOf(cell))))]
    .sort((a, b) => budgetRank(a) - budgetRank(b));
  setOptions(byId(view.controlId), controls, view.controlDefault, view.controlLabel);

  const families = [...new Set(rows.map((row) => DATA.models[row.model_id].category))].sort();
  const groups = ["top", "all", ...families];
  const topLabel = view.leaders === "reference" ? `Top 6 at ${view.referenceBudget} samples` : "Top 6 by mean Elo";
  setOptions(byId(`${prefix}-group`), groups, "top",
    (group) => group === "top" ? topLabel : group === "all" ? "All models" : group);

  [`${prefix}-domain`, view.controlId, `${prefix}-group`, `${prefix}-ci`].forEach((id) => byId(id).addEventListener("change", () => renderBudget(prefix)));
  renderBudget(prefix);
}

function preferTunedRows(rows) {
  if (SHOW_ALL_MODELS) return rows;
  const parents = new Set(rows.map((row) => DATA.models[row.model_id].tuned_from).filter(Boolean));
  return rows.filter((row) => !parents.has(row.model_id));
}

function renderBudget(prefix) {
  const view = BUDGET_VIEWS[prefix];
  const mobile = isMobileViewport();
  const domain = byId(`${prefix}-domain`).value;
  const control = budgetValue(byId(view.controlId).value);
  const group = byId(`${prefix}-group`).value;
  const showCI = byId(`${prefix}-ci`).checked;
  const cells = DATA.cell_options.filter((cell) =>
    sameBudget(view.controlOf(cell), control) && view.includes(cell)
  );
  const cellIds = new Set(cells.map((cell) => cell.id));
  const budgetByCell = Object.fromEntries(cells.map((cell) => [cell.id, budgetValue(view.axisOf(cell))]));
  let rows = DATA.domain_elo.filter((row) =>
    row.metric === "f1_macro" && row.domain === domain && cellIds.has(row.cell) && !EXCLUDED.has(row.model_id) && (SHOW_ALL_MODELS || row.model_id !== "AUTOGLUON")
  );
  rows = preferTunedRows(rows);
  if (!rows.length) return;

  // Use the 100-sample reference ranking for sample-budget curves; feature-budget
  // leaders use mean Elo across the budgets where each model is rated.
  const ranked = view.leaders === "reference"
    ? rows.filter((row) => sameBudget(budgetByCell[row.cell], view.referenceBudget)).map((row) => [row.model_id, row.Elo])
    : [...rows.reduce((totals, row) => {
        const entry = totals.get(row.model_id) || [0, 0];
        return totals.set(row.model_id, [entry[0] + row.Elo, entry[1] + 1]);
      }, new Map())].map(([modelId, [total, count]]) => [modelId, total / count]);
  const topIds = ranked.sort((a, b) => b[1] - a[1]).slice(0, 6).map(([modelId]) => modelId);
  if (group === "top") rows = rows.filter((row) => topIds.includes(row.model_id));
  else if (group !== "all") rows = rows.filter((row) => DATA.models[row.model_id].category === group);

  const modelIds = [...new Set(rows.map((row) => row.model_id))]
    .sort((a, b) => {
      const aRank = topIds.indexOf(a), bRank = topIds.indexOf(b);
      if (aRank < 0 && bRank < 0) return DATA.models[a].display.localeCompare(DATA.models[b].display);
      if (aRank < 0) return 1;
      if (bRank < 0) return -1;
      return aRank - bRank;
    });
  const ticks = [...new Set(rows.map((row) => budgetByCell[row.cell]))].sort((a, b) => budgetRank(a) - budgetRank(b));
  const categorical = view.axisType === "category";
  const position = (value) => categorical ? view.axisLabel(value) : value;
  const beyondLimit = [];
  const traces = modelIds.map((modelId, index) => {
    const points = rows
      .filter((row) => row.model_id === modelId)
      .sort((a, b) => budgetRank(budgetByCell[a.cell]) - budgetRank(budgetByCell[b.cell]));
    const meta = DATA.models[modelId];
    const symbol = SYMBOLS[index % SYMBOLS.length];
    const above = points.map((row) => aboveRegularFeatureLimit(row, budgetByCell[row.cell]));
    if (categorical && above.some(Boolean)) beyondLimit.push(meta.display);
    return {
      type: "scatter",
      mode: "lines+markers",
      name: modelLabel(meta),
      x: points.map((row) => position(budgetByCell[row.cell])),
      y: points.map((row) => row.Elo),
      error_y: {
        visible: showCI,
        type: "data",
        symmetric: false,
        array: points.map((row) => row.Elo < 0 ? null : nonnegativeEloInterval(row)?.plus),
        arrayminus: points.map((row) => row.Elo < 0 ? null : nonnegativeEloInterval(row)?.minus),
        color: meta.color,
        thickness: 0.8,
        width: 2,
      },
      customdata: points.map((row) => [row.Elo_lo, row.Elo_hi, row.n_targets, meta.category]),
      line: { color: meta.color, width: 3.2, dash: DASHES[index % DASHES.length] },
      marker: {
        color: meta.color,
        size: 7,
        symbol: categorical ? above.map((value) => value ? `${symbol}-open` : symbol) : symbol,
        line: { color: categorical ? meta.color : css("--surface"), width: categorical ? 2 : 1 },
      },
      hovertemplate: `<b>${modelLabel(meta)}</b>${overlapTooltip(meta)}<br>${view.hover} %{x}<br>Elo %{y:.0f}<br>95% interval [%{customdata[0]:.0f}, %{customdata[1]:.0f}]<br>%{customdata[2]} targets · %{customdata[3]}<extra></extra>`,
    };
  });

  if (showCI) modelIds.forEach((modelId) => {
    const crossing = rows.filter((row) => row.model_id === modelId && row.Elo < 0 && nonnegativeEloInterval(row));
    if (!crossing.length) return;
    traces.push({
      type: "scatter", mode: "markers", showlegend: false, hoverinfo: "skip",
      x: crossing.map((row) => position(budgetByCell[row.cell])), y: crossing.map(() => 0),
      marker: { size: 0 },
      error_y: {
        type: "data", symmetric: false, array: crossing.map((row) => nonnegativeEloInterval(row).plus),
        arrayminus: crossing.map(() => 0), color: DATA.models[modelId].color, thickness: 0.8, width: 2,
      },
    });
  });
  const yLow = Math.min(...rows.map((row) => showCI ? Math.min(row.Elo, Math.max(0, row.Elo_lo)) : row.Elo));
  const yHigh = Math.max(...rows.map((row) => showCI ? row.Elo_hi : row.Elo));
  const yPadding = Math.max(25, (yHigh - yLow) * 0.08);
  const yRange = [Math.floor((yLow - yPadding) / 25) * 25, Math.ceil((yHigh + yPadding) / 25) * 25];
  const xaxis = categorical
    ? axes({ title: view.axisTitle, type: "category", categoryorder: "array", categoryarray: ticks.map(position) })
    : axes({ title: view.axisTitle, type: "log", tickvals: ticks, ticktext: ticks.map(view.axisLabel) });
  const layout = baseLayout({
    height: mobile ? 680 : 560,
    margin: { l: mobile ? 58 : 70, r: prefix === "feat" ? 10 : mobile ? 16 : 25, t: 22, b: mobile ? 150 : 70 },
    xaxis,
    yaxis: axes({ title: "Bradley–Terry Elo (Random Forest = 1,000)", range: yRange }),
    legend: { orientation: "h", x: 0, y: mobile ? -0.3 : -0.2, font: { size: mobile ? 11 : 12, color: css("--muted") } },
  });
  Plotly.react(byId(view.chart), traces, layout, PLOT_CONFIG);
  const intervalNote = showCI ? "target-bootstrap 95% intervals clipped at zero; original bounds available on hover" : "95% intervals available on hover";
  const limitNote = beyondLimit.length
    ? ` · open markers: beyond the regular feature limit of ${beyondLimit.join(", ")}`
    : "";
  const selectionNote = group === "top" && view.leaders === "reference"
    ? ` · top six models ranked at ${view.referenceBudget} training samples`
    : "";
  const poolNote = prefix === "perf"
    ? " Targets can differ across budgets; the paper uses a fixed shared target pool."
    : "";
  byId(view.note).textContent = `${domainDescription(domain)} · ${view.describe(control)} · Elo at each available ${view.budgetNote}${selectionNote}${limitNote} · ${intervalNote}. Tuned versions replace their untuned counterparts where available.${poolNote}`;
}

function initializeCost() {
  initializeCostView("cost");
  initializeCostView("prediction-cost");
}

function initializeCostView(prefix) {
  const metricSelect = byId(`${prefix}-metric`);
  const domainSelect = byId(`${prefix}-domain`);
  const capSelect = byId(`${prefix}-cap`);
  const sampleSelect = byId(`${prefix}-samples`);
  const reference = DATA.cell_options.find((cell) => cell.id === DATA.meta.reference_cell) || DATA.cell_options[0];
  const caps = [...new Set(DATA.cell_options.map((cell) => cell.feature_cap))].sort((a, b) => Number(a) - Number(b));
  const domains = DATA.domains.filter((domain) => DATA.cost_grid.some((row) => row.domain === domain));
  setOptions(metricSelect, DATA.elo_metrics, "f1_macro", (metric) => ELO_METRICS[metric]);
  setOptions(domainSelect, domains, "all", domainLabel);
  setOptions(capSelect, caps, reference.feature_cap, capLabel);

  function updateSamples(preferred) {
    const cap = budgetValue(capSelect.value);
    const samples = DATA.cell_options
      .filter((cell) => sameBudget(cell.feature_cap, cap))
      .map((cell) => cell.n_train)
      .sort((a, b) => Number(a) - Number(b));
    setOptions(sampleSelect, samples, preferred, sampleLabel);
  }

  metricSelect.addEventListener("change", () => renderCost(prefix));
  domainSelect.addEventListener("change", () => renderCost(prefix));
  capSelect.addEventListener("change", () => { updateSamples(sampleSelect.value); renderCost(prefix); });
  sampleSelect.addEventListener("change", () => renderCost(prefix));
  updateSamples(reference.n_train);
  renderCost(prefix);
}

function paretoFrontier(rows, metric, timeColumn) {
  return rows.filter((candidate) => !rows.some((other) =>
    other.model_id !== candidate.model_id
    && other[timeColumn] <= candidate[timeColumn]
    && other[metric] >= candidate[metric]
    && (other[timeColumn] < candidate[timeColumn] || other[metric] > candidate[metric])
  )).sort((a, b) => a[timeColumn] - b[timeColumn]);
}

function costLabelAnnotations(rows, metric, timeColumn, mobile, width, height, xRange, yRange) {
  const fontSize = mobile ? 10 : 11;
  const context = typeof document.createElement === "function" ? document.createElement("canvas").getContext("2d") : null;
  if (context) context.font = fontSize + 'px "Atkinson Hyperlegible Next", sans-serif';
  const points = rows.map((row) => ({
    row, x: width * (Math.log10(row[timeColumn]) - xRange[0]) / (xRange[1] - xRange[0]),
    y: height * (yRange[1] - row[metric]) / (yRange[1] - yRange[0]),
    w: Math.min(width - 8, (context ? context.measureText(modelLabel(row)).width : modelLabel(row).length * fontSize * 0.65) + 10),
  }));
  const occupied = [];
  const overlaps = (a, b) => a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1];
  // Place wide labels first; search the actual plot geometry instead of dropping labels.
  return [...points].sort((a, b) => b.w - a.w || a.y - b.y).map((point) => {
    let best, bestScore = Infinity;
    const h = fontSize + 8;
    for (let y = h / 2 + 4; y <= height - h / 2 - 4; y += h + 3) {
      for (let x = point.w / 2 + 4; x <= width - point.w / 2 - 4; x += 8) {
        const box = [x - point.w / 2, y - h / 2, x + point.w / 2, y + h / 2];
        if (occupied.some((other) => overlaps(box, other))) continue;
        const covered = points.filter((p) => overlaps(box, [p.x - 8, p.y - 8, p.x + 8, p.y + 8])).length;
        const score = (x - point.x) ** 2 + (y - point.y) ** 2 + covered * 1000000;
        if (score < bestScore) { bestScore = score; best = { x, y, box }; }
      }
    }
    if (!best) throw new Error("Insufficient space for timing labels");
    occupied.push(best.box);
    return {
      x: Math.log10(point.row[timeColumn]), y: point.row[metric], xref: "x", yref: "y",
      name: point.row.model_id, text: modelLabel(point.row), showarrow: true, arrowhead: 0, arrowwidth: 0.7,
      arrowcolor: css("--muted"), standoff: 7,
      ax: best.x - point.x, ay: best.y - point.y, axref: "pixel", ayref: "pixel",
      xanchor: "center", yanchor: "middle", borderpad: 2,
      bgcolor: css("--surface"), font: { size: fontSize, color: css("--ink") },
      captureevents: false,
    };
  });
}

function costPlotSpec(rows, metric, timing, mobile, chartWidth = mobile ? 360 : 1100) {
  const metricLabel = ELO_METRICS[metric];
  const categories = [...new Set(rows.map((row) => row.category))];
  const frontier = paretoFrontier(rows, metric, timing.column);
  const frontierIds = new Set(frontier.map((row) => row.model_id));
  const margin = { l: mobile ? 58 : 78, r: mobile ? 16 : 28, t: 24, b: mobile ? 165 : 110, autoexpand: false };
  const width = Math.max(180, chartWidth - margin.l - margin.r);
  const height = Math.max(420, rows.length * (mobile ? 26 : 23) + 24);
  const logs = rows.map((row) => Math.log10(row[timing.column]));
  const values = rows.map((row) => row[metric]);
  const xPad = Math.max(0.15, (Math.max(...logs) - Math.min(...logs)) * 0.12);
  const yPad = Math.max(0.01, (Math.max(...values) - Math.min(...values)) * 0.15);
  const xRange = [Math.min(...logs) - xPad, Math.max(...logs) + xPad];
  const yRange = [Math.min(...values) - yPad, Math.max(...values) + yPad];
  const frontierTrace = {
    type: "scatter",
    mode: "lines+markers",
    name: "Pareto frontier",
    x: frontier.map((row) => row[timing.column]),
    y: frontier.map((row) => row[metric]),
    line: { color: css("--ink"), width: 2, dash: "dot" },
    marker: { color: css("--surface"), size: 8, line: { color: css("--ink"), width: 1.5 } },
    customdata: frontier.map((row) => modelLabel(row) + overlapTooltip(row)),
    hovertemplate: `<b>%{customdata}</b><br>Pareto frontier<br>%{x:.1f} s per fold<br>${metricLabel} %{y:.3f}<extra></extra>`,
  };
  const categoryTraces = categories.map((category) => {
    const points = rows.filter((row) => row.category === category);
    return {
      type: "scatter",
      mode: "markers",
      name: category,
      ids: points.map((row) => row.model_id),
      x: points.map((row) => row[timing.column]),
      y: points.map((row) => row[metric]),
      cliponaxis: false,
      marker: { color: points[0].color, size: points.map((row) => frontierIds.has(row.model_id) ? 12 : 9), opacity: 0.88, line: { color: css("--surface"), width: 1 } },
      customdata: points.map((row) => modelLabel(row) + overlapTooltip(row)),
      hovertemplate: `<b>%{customdata}</b><br>%{x:.1f} s per fold<br>${metricLabel} %{y:.3f}<extra></extra>`,
    };
  });
  const layout = baseLayout({
    height: height + margin.t + margin.b,
    margin,
    meta: { cost: { rows, metric, timing } },
    annotations: costLabelAnnotations(rows, metric, timing.column, mobile, width, height, xRange, yRange),
    xaxis: axes({ title: timing.axis, type: "log", range: xRange, automargin: false }),
    yaxis: axes({ title: `Mean ${metricLabel}`, range: yRange, automargin: false }),
    legend: { orientation: "h", x: 0, y: -75 / height, yanchor: "top", font: { size: mobile ? 11 : 12, color: css("--muted") } },
  });
  return { traces: [frontierTrace, ...categoryTraces], layout };
}

function renderCost(prefix) {
  const mobile = isMobileViewport();
  const timing = COST_VIEWS[prefix];
  const metric = byId(`${prefix}-metric`).value;
  const domain = byId(`${prefix}-domain`).value;
  const cap = budgetValue(byId(`${prefix}-cap`).value);
  const samples = budgetValue(byId(`${prefix}-samples`).value);
  const option = DATA.cell_options.find((cell) => sameBudget(cell.feature_cap, cap) && sameBudget(cell.n_train, samples));
  if (!option) return;
  const rows = DATA.cost_grid.filter((row) =>
    row.cell === option.id && row.domain === domain && !EXCLUDED.has(row.model_id) && (SHOW_ALL_MODELS || row.model_id !== "AUTOGLUON")
  );
  if (!rows.length) return;
  const chart = byId(timing.chart);
  const spec = costPlotSpec(rows, metric, timing, mobile, chart.clientWidth || chart.closest(".panel").clientWidth);
  Plotly.react(chart, spec.traces, spec.layout, PLOT_CONFIG).then(() => {
    chart.removeAllListeners("plotly_restyle");
    chart.on("plotly_restyle", () => {
      const visible = new Set(chart.data.filter((trace) => trace.visible !== "legendonly" && trace.visible !== false).flatMap((trace) => trace.ids || []));
      Plotly.relayout(chart, { annotations: chart.layout.annotations.map((label) => ({ ...label, visible: visible.has(label.name) })) });
    });
  });
}

function initializeRank() {
  const orderedCells = DATA.cell_options.map((cell) => cell.id);
  RANK_REFERENCE_CELL = orderedCells.includes(DATA.meta.reference_cell)
    ? DATA.meta.reference_cell
    : orderedCells[0];
  setOptions(byId("rank-reference"), orderedCells, RANK_REFERENCE_CELL, cellShortLabel);
  byId("rank-reference").addEventListener("change", (event) => {
    RANK_REFERENCE_CELL = event.target.value;
    renderRank();
  });
  renderRank();
}

function visibleRankCorrelation(leftCell, rightCell) {
  const rows = DATA.domain_elo.filter(row => row.domain === "all" && row.metric === "f1_macro" && !EXCLUDED.has(row.model_id) && (SHOW_ALL_MODELS || row.model_id !== "AUTOGLUON"));
  const left = new Map(rows.filter(row => row.cell === leftCell).map(row => [row.model_id, row.Elo]));
  const right = new Map(rows.filter(row => row.cell === rightCell).map(row => [row.model_id, row.Elo]));
  const ids = [...left.keys()].filter(id => right.has(id));
  if (ids.length < 2) return null;
  const ranks = values => values.map(value => 1 + values.filter(other => other < value).length + (values.filter(other => other === value).length - 1) / 2);
  const a = ranks(ids.map(id => left.get(id))), b = ranks(ids.map(id => right.get(id)));
  const mean = (ids.length + 1) / 2;
  const covariance = a.reduce((sum, value, i) => sum + (value - mean) * (b[i] - mean), 0);
  const variance = values => values.reduce((sum, value) => sum + (value - mean) ** 2, 0);
  const denominator = Math.sqrt(variance(a) * variance(b));
  return denominator ? covariance / denominator : null;
}

function renderRank() {
  const mobile = isMobileViewport();
  const { cells } = DATA.rank_correlations;
  const indexByCell = new Map(cells.map((cell, index) => [cell, index]));
  const selectedCell = indexByCell.has(RANK_REFERENCE_CELL)
    ? RANK_REFERENCE_CELL
    : DATA.meta.reference_cell;
  const selectedOption = DATA.cell_options.find((cell) => cell.id === selectedCell);
  const featureCaps = [...new Set(DATA.cell_options.map((cell) => cell.feature_cap))]
    .sort((a, b) => budgetSortValue(a) - budgetSortValue(b));
  const sampleBudgets = [...new Set(DATA.cell_options.map((cell) => cell.n_train))]
    .sort((a, b) => budgetSortValue(a) - budgetSortValue(b));
  const cellAt = (cap, samples) => DATA.cell_options.find(
    (cell) => sameBudget(cell.feature_cap, cap) && sameBudget(cell.n_train, samples)
  );
  const targets = featureCaps.map((cap) => sampleBudgets.map((samples) => cellAt(cap, samples)));
  const values = targets.map((row) => row.map((target) => {
    const targetIndex = target ? indexByCell.get(target.id) : undefined;
    return targetIndex === undefined ? null : visibleRankCorrelation(selectedCell, target.id);
  }));
  const customdata = targets.map((row) => row.map((target) => [target?.id || "", target?.label || ""]));
  const xLabels = sampleBudgets.map(sampleLabel);
  const yLabels = featureCaps.map(capLabel);
  const trace = {
    type: "heatmap", z: values, x: xLabels, y: yLabels, customdata,
    zmin: -1, zmax: 1,
    colorscale: [[0, "#33205e"], [0.35, "#3d6b79"], [0.7, "#69ad87"], [1, "#e6ef83"]],
    colorbar: { thickness: 10, len: 0.75, title: { text: "Spearman ρ", side: "right" } },
    hovertemplate: `<b>%{customdata[1]}</b><br>vs ${escapeHtml(cellShortLabel(selectedCell))}<br>Spearman ρ=%{z:.2f}<extra></extra>`,
  };
  const selectedMarker = {
    type: "scatter", mode: "markers", showlegend: false, hoverinfo: "skip",
    x: [sampleLabel(selectedOption.n_train)], y: [capLabel(selectedOption.feature_cap)],
    marker: { symbol: "square-open", size: mobile ? 30 : 38, color: css("--ink"), line: { width: 3, color: css("--ink") } },
  };
  const layout = baseLayout({
    height: mobile ? 500 : 460,
    margin: { l: mobile ? 68 : 76, r: mobile ? 28 : 58, t: 24, b: mobile ? 78 : 72 },
    xaxis: axes({
      title: "Training samples", type: "category", categoryorder: "array", categoryarray: xLabels,
      showgrid: false, tickfont: { size: mobile ? 11 : 12, color: css("--muted") },
    }),
    yaxis: axes({
      title: "Feature cap", type: "category", categoryorder: "array", categoryarray: yLabels,
      autorange: "reversed", showgrid: false, tickfont: { size: mobile ? 11 : 12, color: css("--muted") },
    }),
  });
  const chart = byId("rank-chart");
  Plotly.react(chart, [trace, selectedMarker], layout, PLOT_CONFIG).then(() => {
    if (chart.dataset.rankClickBound === "true") return;
    chart.on("plotly_click", (event) => {
      const targetCell = event.points?.[0]?.customdata?.[0];
      if (!targetCell || !indexByCell.has(targetCell)) return;
      RANK_REFERENCE_CELL = targetCell;
      byId("rank-reference").value = targetCell;
      renderRank();
    });
    chart.dataset.rankClickBound = "true";
  });
  byId("rank-note").textContent = `Compared with ${cellShortLabel(selectedCell)}. Click a square or choose another reference cell; exact values are available on hover.`;
}

function renderComposition() {
  const mobile = isMobileViewport();
  const tasks = [...new Set(DATA.datasets.map((dataset) => dataset.task))].sort();
  const modalities = [...new Set(DATA.datasets.map((dataset) => dataset.modality))]
    .sort((a, b) => {
      const count = (modality) => DATA.datasets.filter((dataset) => dataset.modality === modality).length;
      return count(a) - count(b) || a.localeCompare(b);
    });
  const colors = [css("--accent"), css("--violet"), css("--amber")];
  const traces = tasks.map((task, index) => ({
    type: "bar",
    orientation: "h",
    name: task,
    y: modalities,
    x: modalities.map((modality) => DATA.datasets.filter(
      (dataset) => dataset.modality === modality && dataset.task === task
    ).length),
    marker: { color: colors[index % colors.length] },
    hovertemplate: `<b>%{y}</b><br>${escapeHtml(task)}: %{x} datasets<extra></extra>`,
  }));
  const layout = baseLayout({
    height: mobile ? 520 : 460,
    margin: { l: mobile ? 105 : 125, r: mobile ? 16 : 25, t: 18, b: mobile ? 100 : 62 },
    barmode: "stack",
    bargap: 0.3,
    xaxis: axes({ title: "Datasets", dtick: 2 }),
    yaxis: axes({ showgrid: false }),
    legend: { orientation: "h", x: 0, y: mobile ? -0.24 : -0.2, font: { size: 11, color: css("--muted") } },
  });
  Plotly.react(byId("composition-chart"), traces, layout, PLOT_CONFIG);
}

function renderRawFiles() {
  byId("raw-files").innerHTML = DATA.raw_exports.map((file) => {
    const unit = file.format === "sqlite3" ? "attempts" : "records";
    const detail = `${number(file.records)} ${unit} · ${bytes(file.bytes)}`;
    const releasesUrl = `${DATA.meta.github_url.replace(/\/$/, "")}/releases/`;
    const path = file.path || (file.format === "sqlite3" ? releasesUrl : "");
    const isReleasesPage = path.replace(/\/$/, "") === releasesUrl.replace(/\/$/, "");
    if (file.available === false && !isReleasesPage) {
      return `
        <div class="data-file data-file-pending">
          <span><strong>${escapeHtml(file.name)}</strong><span>${detail} · release upload pending</span></span>
          <b aria-hidden="true">…</b>
        </div>`;
    }
    return `
      <a class="data-file" href="${escapeHtml(path)}"${isReleasesPage ? "" : " download"}>
        <span><strong>${escapeHtml(file.name)}</strong><span>${detail}</span></span>
        <b aria-hidden="true">${isReleasesPage ? "↗" : "↓"}</b>
      </a>`;
  }).join("");
}

function renderCharts() {
  renderElo();
  renderBudget("perf");
  renderBudget("feat");
  renderCost("cost");
  renderCost("prediction-cost");
  renderRank();
  renderComposition();
}

const ANALYSIS_VIEWS = ["strict", "adaptive", "conditional"];

function availableAnalysisViews() {
  return ANALYSIS_VIEWS.filter((name) => DATA.analysis_views[name]);
}

function selectAnalysisView(view) {
  const selected = DATA.analysis_views[view];
  if (!selected) throw new Error(`Unknown analysis view: ${view}`);
  Object.entries(selected).forEach(([field, value]) => { DATA[field] = value; });
  availableAnalysisViews().forEach((name) => {
    const active = name === view;
    const button = byId(`view-${name}`);
    button.setAttribute("aria-pressed", String(active));
    button.classList.toggle("button-primary", active);
    const note = byId(`analysis-view-note-${name}`);
    note.classList.toggle("is-active", active);
    note.setAttribute("aria-hidden", String(!active));
  });
  renderReferencePodium();
  if (byId("model-card-model")) renderModelCard();
  if (CHARTS_READY) renderCharts();
}

function initializeAnalysisView() {
  ANALYSIS_VIEWS.forEach((name) => {
    const button = byId(`view-${name}`);
    if (!DATA.analysis_views[name]) {
      // A published dashboard may predate a view; .button sets an explicit display, so
      // the hidden attribute alone would not remove the control.
      button.hidden = true;
      button.style.display = "none";
      return;
    }
    button.addEventListener("click", () => selectAnalysisView(name));
  });
  selectAnalysisView("strict");
}

function updateThemeButton() {
  const button = byId("theme-toggle");
  const dark = document.body.classList.contains("dark-theme");
  const label = dark ? "Switch to light theme" : "Switch to dark theme";
  button.setAttribute("aria-label", label);
  button.setAttribute("aria-pressed", String(dark));
  button.title = label;
}

function initializeTheme() {
  const stored = localStorage.getItem("tabbench-bio-theme");
  const dark = stored === "dark" || (!stored && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.body.classList.toggle("dark-theme", dark);
  updateThemeButton();
  byId("theme-toggle").addEventListener("click", () => {
    document.body.classList.toggle("dark-theme");
    localStorage.setItem("tabbench-bio-theme", document.body.classList.contains("dark-theme") ? "dark" : "light");
    updateThemeButton();
    if (CHARTS_READY && document.body.dataset.page === "home") renderCharts();
    if (document.body.dataset.page === "dataset") renderDatasetCharts();
  });
}

function showLoadError(error) {
  const message = byId("load-error");
  message.hidden = false;
  message.textContent = `${error.message}. Serve the site directory over HTTP.`;
  console.error(error);
}

function initializeModelVisibility() {
  const toggle = byId("show-all-models");
  if (!toggle) return;
  toggle.setAttribute("aria-pressed", String(SHOW_ALL_MODELS));
  toggle.classList.toggle("button-primary", SHOW_ALL_MODELS);
  toggle.addEventListener("click", () => {
    SHOW_ALL_MODELS = !SHOW_ALL_MODELS;
    toggle.setAttribute("aria-pressed", String(SHOW_ALL_MODELS));
    toggle.classList.toggle("button-primary", SHOW_ALL_MODELS);
    EXCLUDED = new Set(SHOW_ALL_MODELS ? [] : DATA.meta.plot_excluded_models);
    const page = document.body.dataset.page;
    if (page === "home") {
      renderReferencePodium();
      renderCharts();
    } else if (page === "models") initializeModelCard();
    else if (page === "dataset") {
      renderDatasetCharts();
    }
  });
}

async function main() {
  initializeTheme();
  const page = document.body.dataset.page;
  if (page === "changelog" || page === "imprint" || page === "citation") return;
  const response = await fetch(page === "datasets" || page === "dataset" ? "data/datasets/index.json" : "data/dashboard.json", { cache: "no-cache" });
  if (!response.ok) throw new Error(`Could not load benchmark data (HTTP ${response.status})`);
  DATA = await response.json();
  EXCLUDED = new Set(SHOW_ALL_MODELS ? [] : DATA.meta.plot_excluded_models);
  initializeModelVisibility();
  if (page === "home") {
    initializeMeta();
    initializeElo();
    initializeBudget("perf");
    initializeBudget("feat");
    initializeCost();
    initializeRank();
    renderComposition();
    initializeAnalysisView();
    const downloadButton = byId("download-all-figures");
    downloadButton.disabled = false;
    downloadButton.addEventListener("click", () => downloadAllFigures(downloadButton));
    CHARTS_READY = true;
    let costResizeTimer;
    window.addEventListener("resize", () => {
      clearTimeout(costResizeTimer);
      costResizeTimer = setTimeout(() => { renderCost("cost"); renderCost("prediction-cost"); }, 120);
    });
    document.querySelectorAll(".panel-collapse").forEach((panel) => {
      const toggle = panel.querySelector(".panel-toggle");
      toggle.addEventListener("click", () => {
        const expanded = toggle.getAttribute("aria-expanded") !== "true";
        toggle.setAttribute("aria-expanded", String(expanded));
        toggle.querySelector(".panel-toggle-label").textContent = expanded ? "×" : "Expand";
        toggle.setAttribute("aria-label", (expanded ? "Close " : "Expand ") + panel.querySelector("h3").textContent);
        panel.classList.toggle("is-expanded", expanded);
        const controls = panel.querySelector(".panel-head .controls");
        if (controls) controls.hidden = !expanded;
        byId(toggle.getAttribute("aria-controls")).hidden = !expanded;
        if (expanded) {
          if (panel.id === "fitting-cost-panel") renderCost("cost");
          else if (panel.id === "prediction-cost-panel") renderCost("prediction-cost");
          else Plotly.Plots.resize(panel.querySelector(".plot"));
        }
      });
    });
  } else if (page === "models") {
    Object.assign(DATA, DATA.analysis_views.strict);
    initializeModelCard();
  } else if (page === "datasets") {
    initializeDatasetExplorer();
  } else if (page === "dataset") {
    await initializeDatasetDetail();
  } else if (page === "artifacts") {
    renderRawFiles();
  } else {
    throw new Error(`Unknown page: ${page}`);
  }
}

main().catch(showLoadError);

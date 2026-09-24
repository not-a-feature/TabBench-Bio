"use strict";

const SOURCE_LABELS = {openml: "OpenML", geo: "GEO", geo_matrix: "GEO", tcga: "TCGA", mgnify: "MGnify", tdc: "TDC", chembl: "ChEMBL", gp: "Genomic prediction", local: "Local", fusionai: "FusionAI", metagenomics: "MetaML"};
const DATASET_FILTERS = ["modality", "task", "source"];
const MODEL_LINE_COLORS = ["#a11e3b", "#168c85", "#b58928", "#8277cc", "#3682b9", "#ce7454", "#71974e", "#c666a3"];
let ACTIVE_DATASET = null;
let DATASET_SCORES = [];

function sourceLabel(source) { return SOURCE_LABELS[source] || source; }
function datasetLink(datasetId) {
  const params = new URLSearchParams(window.location.search);
  params.set("id", datasetId);
  return `dataset.html?${params}`;
}

function filteredDatasetRows() {
  const query = byId("dataset-search").value.trim().toLowerCase();
  const sort = byId("dataset-sort").value;
  const field = sort === "name-desc" ? "display_name" : sort;
  return DATA.datasets.filter(row =>
    [row.display_name, row.dataset_id, row.target, row.modality, row.task, sourceLabel(row.source)].join(" ").toLowerCase().includes(query)
    && DATASET_FILTERS.every(key => !byId(`dataset-${key}`).value || (key === "source" ? sourceLabel(row.source) : row[key]) === byId(`dataset-${key}`).value)
  ).sort((a, b) => {
    const left = field === "source" ? sourceLabel(a.source) : a[field];
    const right = field === "source" ? sourceLabel(b.source) : b[field];
    return (sort === "name-desc" ? -1 : 1) * left.localeCompare(right, undefined, {numeric: true})
      || a.display_name.localeCompare(b.display_name);
  });
}

function renderDatasetRegistry() {
  const params = new URLSearchParams();
  if (byId("dataset-search").value) params.set("q", byId("dataset-search").value);
  DATASET_FILTERS.forEach(key => { if (byId(`dataset-${key}`).value) params.set(key, byId(`dataset-${key}`).value); });
  if (byId("dataset-sort").value !== "display_name") params.set("sort", byId("dataset-sort").value);
  history.replaceState(null, "", `${location.pathname}${params.size ? `?${params}` : ""}`);
  const rows = filteredDatasetRows();
  byId("dataset-result-count").textContent = `${rows.length} of ${DATA.datasets.length} datasets`;
  byId("dataset-table").innerHTML = rows.length ? rows.map(row => `<tr>
    <td><a class="dataset-name dataset-link" href="${escapeHtml(datasetLink(row.dataset_id))}">${escapeHtml(row.display_name)}</a><span class="dataset-id">${escapeHtml(row.dataset_id)}</span></td>
    <td>${escapeHtml(row.modality)}</td><td>${escapeHtml(row.target)}</td><td>${escapeHtml(row.task)}</td>
    <td>${row.source_url ? `<a class="source-link" href="${escapeHtml(row.source_url)}" target="_blank" rel="noreferrer">${escapeHtml(sourceLabel(row.source))} ↗</a>` : escapeHtml(sourceLabel(row.source))}</td>
    <td><a class="dataset-open" href="${escapeHtml(datasetLink(row.dataset_id))}" aria-label="Explore ${escapeHtml(row.display_name)}">Explore →</a></td>
  </tr>`).join("") : '<tr><td colspan="6" class="dataset-empty">No datasets match these filters. Clear filters to see the full registry.</td></tr>';
}

function initializeDatasetExplorer() {
  const params = new URLSearchParams(location.search);
  byId("dataset-search").value = params.get("q") || "";
  DATASET_FILTERS.forEach(key => {
    const values = [...new Set(DATA.datasets.map(row => key === "source" ? sourceLabel(row.source) : row[key]))].sort();
    setOptions(byId(`dataset-${key}`), ["", ...values], params.get(key) || "", value => value || "All");
    byId(`dataset-${key}`).addEventListener("change", renderDatasetRegistry);
  });
  setOptions(byId("dataset-sort"), ["display_name", "name-desc", "modality", "task", "source"], params.get("sort") || "display_name", value => ({display_name: "Name A–Z", "name-desc": "Name Z–A", modality: "Modality", task: "Task", source: "Source"})[value]);
  byId("dataset-sort").addEventListener("change", renderDatasetRegistry);
  byId("dataset-search").addEventListener("input", renderDatasetRegistry);
  byId("dataset-reset").addEventListener("click", () => {
    byId("dataset-search").value = "";
    DATASET_FILTERS.forEach(key => { byId(`dataset-${key}`).value = ""; });
    byId("dataset-sort").value = "display_name";
    renderDatasetRegistry();
  });
  renderDatasetRegistry();
}

function selectedDatasetMetric() {
  return ACTIVE_DATASET.metrics.find(metric => metric.key === byId("detail-metric").value);
}
function datasetCells() {
  const available = new Set(DATASET_SCORES.map(row => row.cell));
  return DATA.cell_options.filter(cell => available.has(cell.id));
}
function updateDatasetSampleOptions() {
  const samples = datasetCells().filter(cell => sameBudget(cell.feature_cap, byId("detail-cap").value)).map(cell => cell.n_train)
    .sort((a, b) => budgetSortValue(a) - budgetSortValue(b));
  setOptions(byId("detail-samples"), samples, byId("detail-samples").value, sampleLabel);
}
function scoreRows(cellId, metric) {
  return DATASET_SCORES.filter(row => row.cell === cellId && !EXCLUDED.has(row.model_id) && Number.isFinite(row.values[metric.key]))
    .sort((a, b) => (metric.better === "low" ? 1 : -1) * (a.values[metric.key] - b.values[metric.key]) || a.model_id.localeCompare(b.model_id));
}
function datasetModelColor(id) {
  if (id === "RF") return css("--ink");
  return MODEL_LINE_COLORS[Object.keys(DATA.models).sort().indexOf(id) % MODEL_LINE_COLORS.length];
}

function renderDatasetCharts() {
  if (!ACTIVE_DATASET || !window.Plotly) return;
  const metric = selectedDatasetMetric();
  const featureCap = budgetValue(byId("detail-cap").value);
  const samples = budgetValue(byId("detail-samples").value);
  const cells = datasetCells().filter(cell => sameBudget(cell.feature_cap, featureCap)).sort((a, b) => budgetSortValue(a.n_train) - budgetSortValue(b.n_train));
  const cell = cells.find(option => sameBudget(option.n_train, samples));
  const rows = cell ? scoreRows(cell.id, metric) : [];
  const params = new URLSearchParams(location.search);
  params.set("metric", metric.key); params.set("p", byId("detail-cap").value); params.set("n", byId("detail-samples").value);
  history.replaceState(null, "", `${location.pathname}?${params}`);
  byId("detail-point").textContent = `${capLabel(featureCap)} features · ${sampleLabel(samples)} training samples · ${metric.better === "low" ? "Lower" : "Higher"} ${metric.label} is better`;
  byId("detail-point-empty").hidden = rows.length > 0;
  const ranked = [...rows].reverse();
  const height = Math.max(380, ranked.length * 28 + 100);
  byId("dataset-ranking-chart").style.height = `${height}px`;
  Plotly.react(byId("dataset-ranking-chart"), [{
    type: "scatter", mode: "markers", x: ranked.map(row => row.values[metric.key]), y: ranked.map(row => DATA.models[row.model_id].display),
    marker: {size: ranked.map(row => row.model_id === "RF" ? 13 : 10), color: ranked.map(row => row.model_id === "RF" ? css("--ink") : css("--accent")), symbol: ranked.map(row => row.model_id === "RF" ? "diamond" : "circle")},
    hovertemplate: `%{y}<br>${metric.label}: %{x:.3f}<extra></extra>`,
  }], baseLayout({height, margin: {l: isMobileViewport() ? 155 : 185, r: 24, t: 20, b: 65}, xaxis: axes({title: `${metric.label} ${metric.better === "low" ? "↓" : "↑"}`}), yaxis: axes({type: "category", categoryorder: "array", categoryarray: ranked.map(row => DATA.models[row.model_id].display), showgrid: false}), showlegend: false}), PLOT_CONFIG);
  const group = byId("detail-models").value;
  const ordering = rows.length ? rows : cells.flatMap(option => scoreRows(option.id, metric));
  const modelIds = group === "top" ? [...new Set([...ordering.slice(0, 5).map(row => row.model_id), "RF"])]
    : Object.keys(DATA.models).filter(id => !EXCLUDED.has(id) && DATASET_SCORES.some(row => row.model_id === id) && (group === "all" || DATA.models[id].category === group));
  const traces = modelIds.map((id, index) => ({
    type: "scatter", mode: "lines+markers", name: DATA.models[id].display,
    x: cells.map(option => sampleLabel(option.n_train)),
    y: cells.map(option => {
      const row = DATASET_SCORES.find(entry => entry.cell === option.id && entry.model_id === id);
      return row ? row.values[metric.key] : null;
    }),
    connectgaps: false, line: {color: datasetModelColor(id), width: id === "RF" ? 3 : 2, dash: id === "RF" ? "dash" : DASHES[index % DASHES.length]},
    marker: {size: 7, symbol: SYMBOLS[index % SYMBOLS.length]},
    hovertemplate: `%{fullData.name}<br>Training samples: %{x}<br>${metric.label}: %{y:.3f}<extra></extra>`,
  }));
  Plotly.react(byId("dataset-budget-chart"), traces, baseLayout({height: 500, margin: {l: 72, r: 24, t: 25, b: 160}, xaxis: axes({title: "Training-sample cap", type: "category", categoryorder: "array", categoryarray: cells.map(option => sampleLabel(option.n_train))}), yaxis: axes({title: `${metric.label} ${metric.better === "low" ? "↓" : "↑"}`}), legend: {orientation: "h", x: 0, y: -0.3, font: {size: 12}}, hovermode: "closest"}), PLOT_CONFIG);
  byId("detail-budget-note").textContent = `${capLabel(featureCap)} features · ${group === "top" ? "Top five at the selected operating point, plus Random Forest." : "Click legend entries to show or hide models."} Full uses all available training samples.`;
  byId("dataset-score-head").innerHTML = `<tr><th>Model</th>${ACTIVE_DATASET.metrics.map(item => `<th>${escapeHtml(item.label)} ${item.better === "low" ? "↓" : "↑"}</th>`).join("")}</tr>`;
  byId("dataset-score-body").innerHTML = rows.map(row => `<tr${row.model_id === "RF" ? ' class="dataset-baseline-row"' : ""}><td>${escapeHtml(DATA.models[row.model_id].display)}${row.model_id === "RF" ? ' <span class="tag">Baseline</span>' : ""}</td>${ACTIVE_DATASET.metrics.map(item => `<td>${Number.isFinite(row.values[item.key]) ? row.values[item.key].toFixed(3) : "—"}</td>`).join("")}</tr>`).join("");
}

async function initializeDatasetDetail() {
  const params = new URLSearchParams(location.search);
  ACTIVE_DATASET = DATA.datasets.find(dataset => dataset.dataset_id === params.get("id"));
  if (!ACTIVE_DATASET) { byId("detail-not-found").hidden = false; byId("dataset-detail-content").hidden = true; return; }
  const back = new URLSearchParams(params);
  ["id", "metric", "p", "n"].forEach(key => back.delete(key));
  byId("dataset-back").href = `datasets.html${back.size ? `?${back}` : ""}`;
  document.title = `${ACTIVE_DATASET.display_name} · TabBench-Bio`;
  byId("detail-title").textContent = ACTIVE_DATASET.display_name;
  byId("detail-id").textContent = ACTIVE_DATASET.dataset_id;
  byId("detail-modality").textContent = ACTIVE_DATASET.modality;
  byId("detail-task").textContent = ACTIVE_DATASET.problem_type === "binary" ? "Binary classification" : ACTIVE_DATASET.task;
  byId("detail-target").textContent = ACTIVE_DATASET.target;
  byId("detail-source").textContent = `${sourceLabel(ACTIVE_DATASET.source)} ↗`;
  byId("detail-source").hidden = !ACTIVE_DATASET.source_url;
  if (ACTIVE_DATASET.source_url) byId("detail-source").href = ACTIVE_DATASET.source_url;
  const response = await fetch(`data/datasets/${ACTIVE_DATASET.scores_file}`);
  if (!response.ok) throw new Error("Could not load dataset scores");
  const payload = await response.json();
  if (payload.dataset_id !== ACTIVE_DATASET.dataset_id) throw new Error("Dataset score identifier mismatch");
  DATASET_SCORES = payload.scores;
  const reference = DATA.cell_options.find(cell => cell.id === ACTIVE_DATASET.default_cell);
  setOptions(byId("detail-metric"), ACTIVE_DATASET.metrics.map(metric => metric.key), params.get("metric") || ACTIVE_DATASET.metrics[0].key, key => ACTIVE_DATASET.metrics.find(metric => metric.key === key).label);
  setOptions(byId("detail-cap"), [...new Set(datasetCells().map(cell => cell.feature_cap))].sort((a, b) => budgetSortValue(a) - budgetSortValue(b)), params.has("p") ? params.get("p") : reference.feature_cap, capLabel);
  setOptions(byId("detail-samples"), datasetCells().filter(cell => sameBudget(cell.feature_cap, byId("detail-cap").value)).map(cell => cell.n_train).sort((a, b) => budgetSortValue(a) - budgetSortValue(b)), params.has("n") ? params.get("n") : reference.n_train, sampleLabel);
  const families = [...new Set(Object.values(DATA.models).filter(model => !EXCLUDED.has(model.id) && model.category !== "AutoML").map(model => model.category))].sort();
  setOptions(byId("detail-models"), ["top", "all", ...families], "top", value => value === "top" ? "Top 5 + Random Forest" : value === "all" ? "All models" : value);
  await loadExternalScript("assets/plotly-cartesian.min.js", "Plotly");
  byId("detail-cap").addEventListener("change", () => { updateDatasetSampleOptions(); renderDatasetCharts(); });
  ["detail-metric", "detail-samples", "detail-models"].forEach(id => byId(id).addEventListener("change", renderDatasetCharts));
  byId("detail-loading").hidden = true;
  renderDatasetCharts();
}

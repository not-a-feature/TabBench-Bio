/* Theme-aware Plotly chart specifications and exports. */

function catColor(c) { return COLORS[c] || COLORS["Other"] || "#9ca3af"; }

function baseLayout(extra) {
  return Object.assign({
    margin: { l: 60, r: 20, t: 16, b: 70 },
    font: { family: "inherit", size: 12 },
    height: 420,
  }, extra || {});
}

function vline(xv, dash, color, width) {
  return {
    type: "line", yref: "paper", y0: 0, y1: 1, x0: xv, x1: xv,
    line: { color: color || "#888", dash: dash || "dash", width: width || 1 },
  };
}

function familyLegend(categories) {
  const seen = [];
  for (const c of categories) if (c && !seen.includes(c)) seen.push(c);
  return seen.map((c) => ({
    type: "scatter", mode: "markers", x: [null], y: [null],
    marker: { color: catColor(c), size: 10 }, name: c, hoverinfo: "skip",
  }));
}

function eloSpec(rows, filename) {
  // AUTOGLUON is shown as a reference line rather than a peer entry.
  const autogluon = rows.find((r) => r.model_id === "AUTOGLUON");
  const peers = rows.filter((r) => r.model_id !== "AUTOGLUON");
  const bar = {
    type: "bar",
    orientation: "h",
    x: peers.map((r) => r.Elo),
    y: peers.map((r) => r.display + (r.clf_only ? " *" : "")),
    marker: { color: peers.map((r) => catColor(r.category)) },
    error_x: {
      type: "data", symmetric: false,
      array: peers.map((r) => r.Elo_hi - r.Elo),
      arrayminus: peers.map((r) => r.Elo - r.Elo_lo),
      color: "#888", thickness: 1, width: 3,
    },
    hovertemplate: "%{y}<br>Elo %{x}<extra></extra>", showlegend: false,
  };
  const shapes = [vline(1000, "dash", "#888", 1)];
  const annotations = [];
  if (autogluon) {
    shapes.push(vline(autogluon.Elo, "solid", catColor(autogluon.category), 2));
    annotations.push({
      x: autogluon.Elo, xanchor: "left", yref: "paper", y: 1, yanchor: "bottom",
      text: "AutoGluon 1 h · " + autogluon.Elo, showarrow: false,
      font: { color: catColor(autogluon.category), size: 11 },
      bgcolor: "rgba(255,255,255,0.75)", borderpad: 2,
    });
  }
  const layout = baseLayout({
    xaxis: { title: "Elo (RF = 1000)", zeroline: false },
    yaxis: { automargin: true },
    shapes, annotations, bargap: 0.3,
    legend: { orientation: "h", y: 1.05, x: 0, font: { size: 11 } },
    height: Math.max(420, peers.length * 28 + 100),
    margin: { l: 135, r: 20, t: 30, b: 70 },
  });
  return { traces: [bar, ...familyLegend(peers.map((r) => r.category))], layout, filename };
}

function specWinrate(fig) {
  const labels = fig.labels, z = fig.matrix, ann = [];
  for (let i = 0; i < labels.length; i++) {
    for (let j = 0; j < labels.length; j++) {
      if (i !== j) ann.push({
        x: labels[j], y: labels[i], text: String(z[i][j]),
        showarrow: false, font: { size: 9, color: "#222" },
      });
    }
  }
  const layout = baseLayout({
    xaxis: { tickangle: -45, automargin: true },
    yaxis: { autorange: "reversed", automargin: true },
    annotations: ann,
    height: Math.max(380, 30 * labels.length + 140),
    margin: { l: 120, r: 20, t: 16, b: 120 },
  });
  return {
    traces: [{
      type: "heatmap", z, x: labels, y: labels, colorscale: "RdYlGn",
      colorbar: { title: "targets won", thickness: 12 },
      hovertemplate: "%{y} vs %{x}: %{z}<extra></extra>",
    }],
    layout, filename: "win_rates",
  };
}

function specPerfTime(fig) {
  const byCat = {};
  for (const p of fig.points) (byCat[p.category] || (byCat[p.category] = [])).push(p);
  const traces = Object.keys(byCat).map((cat) => {
    const pts = byCat[cat];
    return {
      type: "scatter", mode: "markers+text", name: cat,
      x: pts.map((p) => p.x), y: pts.map((p) => p.y),
      text: pts.map((p) => p.display), textposition: "top center", textfont: { size: 9 },
      marker: { color: catColor(cat), size: 11 },
      hovertemplate: "%{text}<br>%{x:.1f}s · score %{y:.3f}<extra></extra>",
    };
  });
  const layout = baseLayout({
    xaxis: { title: "Mean training time (s, log)", type: "log", automargin: true },
    yaxis: { title: "Overall Score", automargin: true },
    legend: { font: { size: 11 } }, height: 480,
  });
  return { traces, layout, filename: "perf_vs_cost" };
}

function specScoreHeatmap(fig) {
  const layout = baseLayout({
    xaxis: { tickangle: -45, automargin: true },
    yaxis: { autorange: "reversed", automargin: true },
    height: Math.max(360, 26 * fig.models.length + 160),
    margin: { l: 130, r: 20, t: 16, b: 130 },
  });
  return {
    traces: [{
      type: "heatmap", z: fig.z, x: fig.datasets, y: fig.models,
      colorscale: "Viridis", zmin: 0, zmax: 1,
      colorbar: { title: "norm. score", thickness: 12 },
      hovertemplate: "%{y}<br>%{x}: %{z:.3f}<extra></extra>",
    }],
    layout, filename: "score_heatmap",
  };
}

function specComposition(fig) {
  const traces = fig.panels.map((p, i) => ({
    type: "pie", labels: p.labels, values: p.values, hole: 0.45,
    domain: { row: 0, column: i },
    title: { text: p.title, font: { size: 12 } },
    textinfo: "label+percent", textposition: "inside",
    insidetextorientation: "horizontal", automargin: true, sort: false,
  }));
  const layout = baseLayout({
    grid: { rows: 1, columns: fig.panels.length }, showlegend: false,
    margin: { l: 10, r: 10, t: 30, b: 20 }, height: 360,
  });
  return { traces, layout, filename: "composition" };
}

function specCharacteristics(fig) {
  const ds = fig.datasets;
  const colors = fig.tasks.map((t) => (/reg/i.test(t) ? "#f59e0b" : "#0ea5e9"));
  const n = fig.panels.length, gap = 0.07, w = (1 - gap * (n - 1)) / n;
  const layout = baseLayout({
    height: Math.max(360, 22 * ds.length + 140), bargap: 0.25, showlegend: true,
    legend: { orientation: "h", y: 1.05, x: 0, font: { size: 11 } },
    margin: { l: 160, r: 20, t: 34, b: 50 },
  });
  const traces = fig.panels.map((p, i) => {
    const sfx = i === 0 ? "" : i + 1, x0 = i * (w + gap);
    layout["xaxis" + sfx] = {
      domain: [x0, x0 + w], title: p.title, type: p.log ? "log" : "linear",
      automargin: true, anchor: "y" + sfx,
    };
    layout["yaxis" + sfx] = {
      domain: [0, 1], type: "category", autorange: "reversed",
      showticklabels: i === 0, anchor: "x" + sfx,
    };
    return {
      type: "bar", orientation: "h", y: ds, x: p.values, marker: { color: colors },
      xaxis: "x" + sfx, yaxis: "y" + sfx, showlegend: false,
      hovertemplate: "%{y}<br>" + p.title + ": %{x}<extra></extra>",
    };
  });
  traces.push(
    { type: "scatter", mode: "markers", x: [null], y: [null], xaxis: "x", yaxis: "y", marker: { color: "#0ea5e9", size: 10 }, name: "Classification", hoverinfo: "skip" },
    { type: "scatter", mode: "markers", x: [null], y: [null], xaxis: "x", yaxis: "y", marker: { color: "#f59e0b", size: 10 }, name: "Regression", hoverinfo: "skip" },
  );
  return { traces, layout, filename: "dataset_characteristics" };
}

// Positioning scatter: our datasets in the sample x feature plane (log-log), coloured by
// biological modality, with a samples = features diagonal. Everything below the diagonal
// is HDLSS. Reference collections (other benchmarks' shapes) are overlaid in grey when
// the payload carries them.
function specSamplesFeatures(fig) {
  const traces = [];
  const ref = fig.reference || {};
  const REF_SYMBOLS = ["circle-open", "square-open", "diamond-open", "triangle-up-open", "x-open"];

  // Reference collections first, so our own datasets draw on top of them.
  Object.keys(ref).forEach((name, i) => {
    const pts = ref[name];
    traces.push({
      type: "scatter", mode: "markers", name,
      x: pts.map((p) => p[1]), y: pts.map((p) => p[0]),
      marker: {
        color: "#9ca3af", size: 6, symbol: REF_SYMBOLS[i % REF_SYMBOLS.length],
        line: { width: 1 },
      },
      opacity: 0.55,
      hovertemplate: name + "<br>%{x} features · %{y} samples<extra></extra>",
    });
  });

  const byMod = {};
  for (const p of fig.points) (byMod[p.modality] || (byMod[p.modality] = [])).push(p);
  for (const mod of Object.keys(byMod)) {
    const pts = byMod[mod];
    traces.push({
      type: "scatter", mode: "markers", name: mod,
      x: pts.map((p) => p.x), y: pts.map((p) => p.y),
      text: pts.map((p) => p.display),
      marker: {
        color: (fig.modality_colors || {})[mod] || "#e11d48", size: 11,
        line: { color: "#fff", width: 1 },
      },
      hovertemplate: "%{text}<br>%{x} features · %{y} samples<extra></extra>",
    });
  }

  // samples = features diagonal, spanning the union of every plotted point.
  const xs = traces.flatMap((t) => t.x), ys = traces.flatMap((t) => t.y);
  const lo = Math.max(1, Math.min(...xs, ...ys)), hi = Math.max(...xs, ...ys);
  traces.push({
    type: "scatter", mode: "lines", x: [lo, hi], y: [lo, hi],
    line: { color: "#888", dash: "dash", width: 1 },
    name: "samples = features", hoverinfo: "skip",
  });

  const layout = baseLayout({
    xaxis: { title: "Features (log)", type: "log", automargin: true },
    yaxis: { title: "Samples (log)", type: "log", automargin: true },
    legend: { font: { size: 11 } },
    annotations: [{
      x: 0.98, y: 0.04, xref: "paper", yref: "paper",
      text: "HDLSS: features > samples", showarrow: false,
      font: { size: 11, color: "#374151" }, xanchor: "right",
    }],
    height: 480,
  });
  return { traces, layout, filename: "samples_vs_features" };
}

const FIG_SPECS = {
  samples_features: specSamplesFeatures,
  winrate: specWinrate,
  perf_time: specPerfTime,
  score_heatmap: specScoreHeatmap,
  composition: specComposition,
  characteristics: specCharacteristics,
};

function subplotGrid(n, ncols, layout, opts) {
  ncols = Math.min(ncols || 4, n) || 1;
  const nrows = Math.ceil(n / ncols);
  const gx = 0.05, gy = 0.12;
  const w = (1 - gx * (ncols - 1)) / ncols, h = (1 - gy * (nrows - 1)) / nrows;
  const items = [];
  for (let i = 0; i < n; i++) {
    const r = Math.floor(i / ncols), c = i % ncols, sfx = i === 0 ? "" : i + 1;
    const x0 = c * (w + gx), y1 = 1 - r * (h + gy), y0 = y1 - h;
    layout["xaxis" + sfx] = Object.assign({ domain: [x0, x0 + w], anchor: "y" + sfx }, opts.xaxis || {});
    layout["yaxis" + sfx] = Object.assign({ domain: [y0, y1], anchor: "x" + sfx }, opts.yaxis || {});
    items.push({ xa: "x" + sfx, ya: "y" + sfx, cx: x0 + w / 2, top: y1 });
  }
  return { items, nrows };
}

function subplotTitles(items, titles) {
  return items.map((it, i) => ({
    text: titles[i], x: it.cx, y: Math.min(1, it.top + 0.015),
    xref: "paper", yref: "paper", xanchor: "center", yanchor: "bottom",
    showarrow: false, font: { size: 9, color: "#374151" },
  }));
}

function surfaceHeatmapSpec(g, ds, metric) {
  const models = g.surface[ds][metric], ids = Object.keys(models);
  const caps = g.feature_caps.map(capLabel), samples = g.sample_sizes.map(sampleLabel);
  let zmin = Infinity, zmax = -Infinity;
  for (const id of ids) {
    for (const row of models[id].z) {
      for (const v of row) if (v !== null && v !== undefined) { if (v < zmin) zmin = v; if (v > zmax) zmax = v; }
    }
  }
  if (!isFinite(zmin) || zmin === zmax) { zmin = 0; zmax = 1; }
  const layout = baseLayout({ height: 0, margin: { l: 42, r: 20, t: 26, b: 40 } });
  const { items, nrows } = subplotGrid(ids.length, 4, layout, {
    xaxis: { type: "category", tickfont: { size: 7 } },
    yaxis: { type: "category", tickfont: { size: 7 } },
  });
  layout.height = Math.max(320, nrows * 200 + 60);
  layout.annotations = subplotTitles(items, ids.map((id) => models[id].display));
  const traces = ids.map((id, i) => ({
    type: "heatmap", z: models[id].z, x: caps, y: samples, zmin, zmax, colorscale: "Viridis",
    xaxis: items[i].xa, yaxis: items[i].ya,
    showscale: i === 0, colorbar: i === 0 ? { thickness: 9, len: 0.9 } : undefined,
    hovertemplate: models[id].display + "<br>cap %{x} · n %{y}: %{z:.3f}<extra></extra>",
  }));
  return { traces, layout, filename: "grid_heatmap_" + ds + "_" + metric };
}

function surfaceCurvesSpec(g, ds, metric) {
  const models = g.surface[ds][metric], ids = Object.keys(models);
  const caps = g.feature_caps, samples = g.sample_sizes.map(sampleLabel);
  const palette = caps.map((c, i) =>
    `hsl(${280 - 230 * (caps.length === 1 ? 0 : i / (caps.length - 1))},70%,45%)`);
  const layout = baseLayout({ height: 0, margin: { l: 46, r: 20, t: 26, b: 44 } });
  const { items, nrows } = subplotGrid(ids.length, 4, layout, {
    xaxis: { type: "category", tickfont: { size: 7 }, tickangle: -45 },
    yaxis: { tickfont: { size: 7 } },
  });
  layout.height = Math.max(320, nrows * 200 + 60);
  layout.annotations = subplotTitles(items, ids.map((id) => models[id].display));
  layout.legend = { title: { text: "feature cap" }, font: { size: 9 } };
  const traces = [];
  ids.forEach((id, i) => {
    const z = models[id].z;
    caps.forEach((cap, ci) => {
      traces.push({
        type: "scatter", mode: "lines+markers", x: samples, y: samples.map((_, si) => z[si][ci]),
        line: { color: palette[ci], width: 1.4 }, marker: { size: 3, color: palette[ci] },
        xaxis: items[i].xa, yaxis: items[i].ya,
        name: capLabel(cap), legendgroup: capLabel(cap), showlegend: i === 0,
        hovertemplate: "cap " + capLabel(cap) + " · n %{x}: %{y:.3f}<extra></extra>",
      });
    });
  });
  return { traces, layout, filename: "grid_curves_" + ds + "_" + metric };
}

const LIVE = []; // { div, spec } for every chart currently on the page

function themeColors(dark) {
  return dark
    ? { fg: "#e5e7eb", paper: "#111827", grid: "#374151" }
    : { fg: "#1f2937", paper: "#ffffff", grid: "#e5e7eb" };
}

// Apply light/dark colours to a (theme-agnostic) layout, returning a fresh copy.
function themedLayout(layout, dark) {
  if (dark === undefined) dark = document.body.classList.contains("dark-theme");
  const c = themeColors(dark);
  const L = structuredClone(layout);
  L.dragmode = false;
  L.paper_bgcolor = c.paper;
  L.plot_bgcolor = c.paper;
  L.font = Object.assign({}, L.font, { color: c.fg });
  for (const k in L) {
    if (k.startsWith("xaxis") || k.startsWith("yaxis")) {
      L[k] = Object.assign({ fixedrange: true, gridcolor: c.grid, linecolor: c.grid, zerolinecolor: c.grid }, L[k]);
    }
  }
  if (Array.isArray(L.annotations)) {
    for (const a of L.annotations) if (a.font && a.font.color === "#374151") a.font.color = c.fg;
  }
  return L;
}

// Modebar config: keep the default PNG export button, add an SVG one.
function plotConfig(filename) {
  return {
    responsive: true, displaylogo: false,
    scrollZoom: false, doubleClick: false,
    modeBarButtonsToRemove: [
      "zoom2d", "pan2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
      "lasso2d", "select2d",
    ],
    toImageButtonOptions: { format: "png", scale: 2, filename },
    modeBarButtonsToAdd: [{
      name: "downloadSvg", title: "Download as SVG", icon: Plotly.Icons.disk,
      click: (gd) => Plotly.downloadImage(gd, { format: "svg", filename }),
    }],
  };
}

function drawSpec(div, spec) {
  for (let i = LIVE.length - 1; i >= 0; i--) if (!document.body.contains(LIVE[i].div)) LIVE.splice(i, 1);
  Plotly.newPlot(div, spec.traces, themedLayout(spec.layout), plotConfig(spec.filename));
  LIVE.push({ div, spec });
}

// Restyle every on-screen chart to the current theme.
function retheme() {
  for (const e of LIVE) {
    if (document.body.contains(e.div)) {
      Plotly.react(e.div, e.spec.traces, themedLayout(e.spec.layout), plotConfig(e.spec.filename));
    }
  }
}

// Render a spec off-screen (always light, for clean export) and return an image blob.
async function specToBlob(spec, ext) {
  const tmp = document.createElement("div");
  tmp.style.cssText = "position:absolute;left:-99999px;top:0;width:1100px;";
  document.body.appendChild(tmp);
  await Plotly.newPlot(tmp, spec.traces, themedLayout(spec.layout, false), { staticPlot: true });
  const uri = await Plotly.toImage(tmp, {
    format: ext, width: 1100, height: spec.layout.height || 500, scale: ext === "png" ? 2 : 1,
  });
  Plotly.purge(tmp);
  tmp.remove();
  return (await fetch(uri)).blob();
}

// Default full-data Elo cell.
const MAIN_ELO_CELL = { cap: "full", samples: "full", domain: "all" };

function mainEloKey(g) {
  return g.default_elo_metric + "|" + MAIN_ELO_CELL.cap + "|" + MAIN_ELO_CELL.samples + "|" + MAIN_ELO_CELL.domain;
}

// Bundle gallery, selected-grid, and complete Elo-cell figures.
async function downloadAllFigures(btn) {
  if (!DATA) return;
  if (typeof JSZip === "undefined") { alert("Could not load JSZip — figure download unavailable."); return; }
  const items = [];
  for (const fig of (DATA.figdata || [])) {
    const build = FIG_SPECS[fig.kind];
    if (build) items.push({ dir: "figures", spec: build(fig), exts: ["svg", "png"] });
  }
  const g = DATA.grid;
  if (g) {
    // Export the default Elo plot independently of the current selection.
    const mainRows = g.elo[mainEloKey(g)];
    if (mainRows && mainRows.length) {
      items.push({ dir: "figures", spec: eloSpec(mainRows, "elo_leaderboard_main"), exts: ["svg", "png"] });
    }
    // The on-screen selection.
    const rows = g.elo[gstate.eloMetric + "|" + gstate.cap + "|" + gstate.samples + "|" + gstate.domain];
    if (rows && rows.length) items.push({ dir: "grid", spec: eloSpec(rows, "elo_leaderboard_" + gstate.eloMetric + "_" + gstate.cap + "_" + gstate.samples + "_" + gstate.domain), exts: ["svg", "png"] });
    const surf = (g.surface[gstate.dataset] || {})[gstate.metric];
    if (surf && Object.keys(surf).length) {
      items.push({ dir: "grid", spec: surfaceHeatmapSpec(g, gstate.dataset, gstate.metric), exts: ["svg", "png"] });
      items.push({ dir: "grid", spec: surfaceCurvesSpec(g, gstate.dataset, gstate.metric), exts: ["svg", "png"] });
    }
    // Every Elo combination the run produced.
    for (const key of Object.keys(g.elo)) {
      const cellRows = g.elo[key];
      if (!cellRows || !cellRows.length) continue;
      items.push({
        dir: "grid/elo_cells",
        spec: eloSpec(cellRows, "elo_" + key.split("|").join("_")),
        exts: ["svg"],
      });
    }
  }
  if (!items.length) return;
  const label = btn ? btn.textContent : null;
  if (btn) btn.disabled = true;
  try {
    const zip = new JSZip();
    const root = zip.folder("tabbench-bio-figures");
    let done = 0;
    for (const { dir, spec, exts } of items) {
      const folder = root.folder(dir);
      for (const ext of exts) folder.file(spec.filename + "." + ext, await specToBlob(spec, ext));
      done++;
      if (btn) btn.textContent = "Preparing… " + done + "/" + items.length;
    }
    const blob = await zip.generateAsync({ type: "blob" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "tabbench-bio-figures.zip";
    a.click();
    URL.revokeObjectURL(a.href);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = label; }
  }
}

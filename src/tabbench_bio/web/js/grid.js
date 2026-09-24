/* Page-level grid sections: the headline per-cell Elo leaderboard and the
 * feature×sample performance-surface browser. */

const gstate = {}; // selections for the page-level grid sections (cell + surface)

function renderGridSections() {
  const g = DATA.grid;
  const eloRoot = document.getElementById("grid-elo");
  const surfRoot = document.getElementById("grid-surface");
  if (!g) {
    if (eloRoot) eloRoot.style.display = "none";
    if (surfRoot) surfRoot.style.display = "none";
    return;
  }
  if (gstate.cap === undefined) {
    gstate.cap = g.default_cap;
    gstate.samples = g.default_samples;
    gstate.domain = g.default_domain;
    gstate.eloMetric = g.default_elo_metric;
    gstate.dataset = g.default_dataset;
    gstate.metric = g.default_metric;
  }
  renderGridElo(g, eloRoot);
  renderGridSurface(g, surfRoot);
}

function renderGridElo(g, root) {
  root.innerHTML = "";
  const hasCell = (cap, samples) => Object.prototype.hasOwnProperty.call(
    g.elo,
    gstate.eloMetric + "|" + cap + "|" + samples + "|" + gstate.domain
  );
  const availableCaps = g.feature_caps.filter((cap) =>
    g.sample_sizes.some((samples) => hasCell(cap, samples))
  );
  if (!availableCaps.includes(gstate.cap)) gstate.cap = availableCaps[0];
  const availableSamples = g.sample_sizes.filter((samples) => hasCell(gstate.cap, samples));
  if (!availableSamples.includes(gstate.samples)) gstate.samples = availableSamples[0];

  const h = document.createElement("h2");
  h.textContent = "🏆 Elo Leaderboard";
  root.appendChild(h);

  const multiDomain = g.domains && g.domains.length > 1;
  const blurb = document.createElement("p");
  blurb.className = "blurb";
  blurb.textContent =
    "Pairwise Elo at a chosen high-dimensional, low-sample-size grid cell — pick a training-set " +
    "size and feature cap" +
    (multiDomain ? ", and optionally restrict to a single dataset domain" : "") +
    ". Computed across the grid datasets with Random Forest = 1000.";
  root.appendChild(blurb);

  const controls = document.createElement("div");
  controls.className = "bd-controls";
  controls.appendChild(
    selectControl("Metric", g.elo_metrics, gstate.eloMetric,
      (v) => { gstate.eloMetric = v; renderGridElo(g, root); }, metricLabel)
  );
  controls.appendChild(
    selectControl("Training samples", availableSamples, gstate.samples,
      (v) => { gstate.samples = v; renderGridElo(g, root); }, sampleLabel)
  );
  controls.appendChild(
    selectControl("Feature cap", availableCaps, gstate.cap,
      (v) => { gstate.cap = v; renderGridElo(g, root); }, capLabel)
  );
  if (g.domains && g.domains.length > 1) {
    controls.appendChild(
      selectControl("Domain", g.domains, gstate.domain,
        (v) => { gstate.domain = v; renderGridElo(g, root); }, domainLabel)
    );
  }
  root.appendChild(controls);

  const rows = g.elo[gstate.eloMetric + "|" + gstate.cap + "|" + gstate.samples + "|" + gstate.domain];
  if (rows && rows.length) {
    const plot = document.createElement("div");
    plot.className = "plot";
    root.appendChild(plot);
    drawSpec(plot, eloSpec(rows, "elo_leaderboard_" + gstate.eloMetric + "_" + gstate.cap + "_" + gstate.samples + "_" + gstate.domain));
    const cap = document.createElement("p");
    cap.className = "blurb";
    cap.textContent =
      "Pairwise Elo across the grid datasets at this cell (Random Forest = 1000), from mean " +
      metricLabel(gstate.eloMetric) + ". Error bars are 95% bootstrap CIs; the dashed line marks " +
      "RF = 1000. When an AutoGluon result exists for the selected cell, its labelled solid " +
      "line is the one-hour AutoML reference. A confidence bound " +
      "may be negative even when the model's point Elo is positive, especially in sparse " +
      "intermediate cells. Bars are coloured by model family. Note: AutoGluon models are fit to optimize " +
      "Macro-F1 (its eval_metric); the metric selector only re-ranks the recorded results, it does " +
      "not refit.";
    root.appendChild(cap);
  } else {
    const e = document.createElement("div");
    e.className = "empty";
    e.textContent = "No grid results for this cell.";
    root.appendChild(e);
  }
}

function renderGridSurface(g, root) {
  root.innerHTML = "";
  if (!g.fig_datasets || !g.fig_datasets.length) { root.style.display = "none"; return; }
  root.style.display = "";

  const h = document.createElement("h2");
  h.textContent = "📉 Performance surface";
  root.appendChild(h);

  const blurb = document.createElement("p");
  blurb.className = "blurb";
  blurb.textContent =
    "How each model's score scales across feature caps and training-set sizes, per dataset. " +
    "Heatmaps read the whole surface; learning curves read scaling with sample size.";
  root.appendChild(blurb);

  const controls = document.createElement("div");
  controls.className = "bd-controls";
  const dsName = (ds) => (g.display_names && g.display_names[ds]) || ds;
  controls.appendChild(
    selectControl("Dataset", g.fig_datasets, gstate.dataset,
      (v) => { gstate.dataset = v; renderGridSurface(g, root); }, dsName)
  );
  controls.appendChild(
    selectControl("Metric", g.fig_metrics, gstate.metric,
      (v) => { gstate.metric = v; renderGridSurface(g, root); }, metricLabel)
  );
  root.appendChild(controls);

  const surf = (g.surface[gstate.dataset] || {})[gstate.metric];
  const figWrap = document.createElement("div");
  figWrap.className = "grid-figs";
  root.appendChild(figWrap); // attach before drawing so Plotly measures the full width
  if (surf && Object.keys(surf).length) {
    for (const build of [surfaceHeatmapSpec, surfaceCurvesSpec]) {
      const plot = document.createElement("div");
      plot.className = "plot";
      figWrap.appendChild(plot);
      drawSpec(plot, build(g, gstate.dataset, gstate.metric));
    }
  } else {
    const e = document.createElement("div");
    e.className = "empty";
    e.textContent = "No surface data for this dataset/metric.";
    figWrap.appendChild(e);
  }
}

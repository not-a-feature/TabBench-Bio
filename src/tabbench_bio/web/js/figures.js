/* Figure gallery: collapsible cards, each drawing a Plotly chart from DATA.figdata. */

function figEmoji(title) {
  if (title.includes("Positioning")) return "🎯";
  if (title.includes("Elo") || title.includes("Rank")) return "🏆";
  if (title.includes("Win")) return "🔢";
  if (title.includes("Cost") || title.includes("Efficiency") || title.includes("Time") || title.includes("Performance")) return "⚡";
  if (title.includes("Heatmap") || title.includes("Scores")) return "📉";
  if (title.includes("Characteristics")) return "📐";
  return "📊";
}

// Render lazily because Plotly needs a laid-out container for sizing.
function figureCard(fig, open, specBuilder) {
  const d = document.createElement("details");
  d.className = "figure";
  d.open = open;
  const s = document.createElement("summary");
  s.textContent = figEmoji(fig.title) + " " + fig.title;
  d.appendChild(s);
  const content = document.createElement("div");
  content.className = "figure-content";
  const plot = document.createElement("div");
  plot.className = "plot";
  const cap = document.createElement("figcaption");
  cap.textContent = fig.caption || "";
  content.append(plot, cap);
  d.appendChild(content);

  let rendered = false;
  const ensure = () => {
    if (rendered) { Plotly.Plots.resize(plot); return; }
    rendered = true;
    drawSpec(plot, specBuilder(fig));
  };
  if (open) requestAnimationFrame(ensure);
  d.addEventListener("toggle", () => { if (d.open) ensure(); });
  return d;
}

function renderFigures() {
  const root = document.getElementById("figures");
  root.innerHTML = "";
  const figs = DATA.figdata || [];
  if (!figs.length) { root.style.display = "none"; return; }
  root.style.display = "";
  const h = document.createElement("h2");
  h.textContent = "Figures";
  root.appendChild(h);

  const openByDefault = new Set(["Benchmark Positioning", "Pairwise Win Rates", "Benchmark Composition"]);
  for (const fig of figs) {
    const build = FIG_SPECS[fig.kind];
    if (!build) continue;
    root.appendChild(figureCard(fig, openByDefault.has(fig.title), build));
  }
}

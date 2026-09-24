/* Entry point: tab bar, payload fetch, and the initial render of every section. */

function buildTabs() {
  const bar = document.getElementById("tabs");
  bar.innerHTML = "";
  for (const t of DATA.order) {
    const b = document.createElement("button");
    b.className = "tab-btn";
    b.textContent = DATA.tabs[t].label;
    b.onclick = () => {
      state.__active = t;
      [...bar.children].forEach((c) => c.classList.remove("active"));
      b.classList.add("active");
      render();
    };
    bar.appendChild(b);
  }
  if (bar.firstChild) bar.firstChild.classList.add("active");
  state.__active = DATA.order[0];
}

async function main() {
  initTheme();
  const dlBtn = document.getElementById("download-all");
  if (dlBtn) dlBtn.onclick = () => downloadAllFigures(dlBtn);
  try {
    const resp = await fetch("data/leaderboard.json", { cache: "no-cache" });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    DATA = await resp.json();
  } catch (e) {
    panel().innerHTML =
      '<div class="empty">Could not load <code>data/leaderboard.json</code> (' + e + ").<br>" +
      "If viewing locally, serve the directory first: <code>python -m http.server</code></div>";
    return;
  }
  COLORS = DATA.category_colors || {};
  if (DATA.meta) {
    if (DATA.meta.title) {
      document.getElementById("title").textContent = "🧬 " + DATA.meta.title;
      document.title = DATA.meta.title;
    }
    if (DATA.meta.subtitle) document.getElementById("subtitle").textContent = DATA.meta.subtitle;
  }
  initState();
  buildTabs();
  render();
  renderGridSections();
  renderFigures();
}

main();

/* Leaderboard tables: the sortable/filterable rank tables and the breakdown matrix. */

function render() {
  const t = state.__active;
  const tab = DATA.tabs[t];
  const st = state[t];
  const root = panel();
  root.innerHTML = "";

  if (tab.kind === "breakdown") { renderBreakdown(t, root); return; }

  const blurb = document.createElement("p");
  blurb.className = "blurb";
  blurb.textContent = tab.blurb || "";
  root.appendChild(blurb);

  if (!tab.rows.length) {
    const e = document.createElement("div");
    e.className = "empty";
    e.textContent = "No results for this tab yet.";
    root.appendChild(e);
    return;
  }

  const toolbar = document.createElement("div");
  toolbar.className = "toolbar";
  for (const f of (tab.filters || [])) {
    const fs = document.createElement("fieldset");
    const lg = document.createElement("legend");
    lg.textContent = f.label;
    fs.appendChild(lg);
    for (const val of distinct(tab.rows, f.column)) {
      const lab = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = !st.filters[f.column].has(val);
      cb.onchange = () => {
        if (cb.checked) st.filters[f.column].delete(val); else st.filters[f.column].add(val);
        render();
      };
      lab.appendChild(cb);
      if (f.column === "Category") {
        const dot = document.createElement("span");
        dot.className = "dot";
        dot.style.background = COLORS[val] || COLORS["Other"] || "#999";
        lab.appendChild(dot);
      }
      lab.appendChild(document.createTextNode(val));
      fs.appendChild(lab);
    }
    toolbar.appendChild(fs);
  }
  const colFs = document.createElement("fieldset");
  const colLg = document.createElement("legend");
  colLg.textContent = "Columns";
  colFs.appendChild(colLg);
  for (const c of tab.columns) {
    if (c.fixed) continue;
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !st.hidden.has(c.key);
    cb.onchange = () => {
      if (cb.checked) st.hidden.delete(c.key); else st.hidden.add(c.key);
      render();
    };
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(c.label));
    colFs.appendChild(lab);
  }
  toolbar.appendChild(colFs);
  root.appendChild(toolbar);

  const rows = tab.rows.filter((r) => passesFilters(r, st, tab));
  const cols = tab.columns.filter((c) => colVisible(c, st));
  const sc = tab.columns.find((c) => c.key === st.sort.key) || cols[0];
  const numeric = sc && (sc.type === "num" || sc.type === "int");
  rows.sort((a, b) => {
    let x = a[sc.key], y = b[sc.key];
    if (x === null || x === undefined || (typeof x === "number" && isNaN(x))) return 1;
    if (y === null || y === undefined || (typeof y === "number" && isNaN(y))) return -1;
    if (numeric) { x = Number(x); y = Number(y); } else { x = String(x); y = String(y); }
    const d = x < y ? -1 : x > y ? 1 : 0;
    return st.sort.dir === "asc" ? d : -d;
  });
  const best = bestValues(cols, rows);

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const htr = document.createElement("tr");
  for (const c of cols) {
    const th = document.createElement("th");
    if (c.type === "model" || c.type === "cat" || c.type === "str") th.className = "txt";
    th.textContent = c.label + " ";
    if (st.sort.key === c.key) {
      const a = document.createElement("span");
      a.className = "arrow";
      a.textContent = st.sort.dir === "asc" ? "▲" : "▼";
      th.appendChild(a);
    }
    th.onclick = () => {
      if (st.sort.key === c.key) st.sort.dir = st.sort.dir === "asc" ? "desc" : "asc";
      else {
        st.sort.key = c.key;
        st.sort.dir = (c.type === "num" || c.type === "int") && c.better !== "low" ? "desc" : "asc";
      }
      render();
    };
    htr.appendChild(th);
  }
  thead.appendChild(htr);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const r of rows) {
    const tr = document.createElement("tr");
    for (const c of cols) {
      const td = document.createElement("td");
      const v = r[c.key];
      if (c.type === "model") {
        td.className = "model";
        td.textContent = fmt(c, v);
      } else if (c.type === "cat") {
        td.className = "cat";
        const dot = document.createElement("span");
        dot.className = "dot";
        dot.style.background = COLORS[v] || COLORS["Other"] || "#999";
        td.appendChild(dot);
        td.appendChild(document.createTextNode(v == null ? "—" : v));
      } else if (c.type === "str") {
        td.className = "txt";
        td.textContent = fmt(c, v);
      } else {
        td.textContent = fmt(c, v);
        if (c.key in best && v !== null && v !== undefined && !isNaN(v) && Number(v) === best[c.key]) {
          td.className = "best";
        }
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  root.appendChild(table);
}

function renderBreakdown(t, root) {
  const tab = DATA.tabs[t];
  const st = state[t];

  const blurb = document.createElement("p");
  blurb.className = "blurb";
  blurb.textContent = tab.blurb || "";
  root.appendChild(blurb);

  const controls = document.createElement("div");
  controls.className = "bd-controls";
  const wrap = document.createElement("label");
  wrap.className = "bd-axis";
  const lab = document.createElement("span");
  lab.className = "bd-axis-label";
  lab.textContent = "Break down by";
  const sel = document.createElement("select");
  for (const a of tab.axes) {
    const opt = document.createElement("option");
    opt.value = a.key; opt.textContent = a.label;
    sel.appendChild(opt);
  }
  sel.value = st.axisKey;
  sel.onchange = () => { st.axisKey = sel.value; build(); };
  wrap.appendChild(lab);
  wrap.appendChild(sel);
  controls.appendChild(wrap);
  root.appendChild(controls);

  const out = document.createElement("div");
  out.className = "bd-output";
  root.appendChild(out);

  function build() {
    out.innerHTML = "";
    const axis = tab.axes.find((a) => a.key === st.axisKey) || tab.axes[0];

    // Derive buckets from the data; axis.options only controls display order.
    const order = axis.options || [];
    const dsCount = {};            // bucket -> # dataset targets
    for (const d of tab.datasets) {
      const b = d[axis.key];
      if (b === null || b === undefined || b === "") continue;
      dsCount[b] = (dsCount[b] || 0) + 1;
    }
    const buckets = Object.keys(dsCount).sort((a, b) => {
      const ia = order.indexOf(a), ib = order.indexOf(b);
      if (ia !== -1 && ib !== -1) return ia - ib;
      if (ia !== -1) return -1;
      if (ib !== -1) return 1;
      return String(a).localeCompare(String(b));
    });

    const cell = {};
    const overall = {};
    for (const d of tab.datasets) {
      const b = d[axis.key];
      if (!buckets.includes(b)) continue;
      for (const mid in d.scores) {
        const v = d.scores[mid];
        if (v === null || v === undefined || isNaN(v)) continue;
        (cell[mid] || (cell[mid] = {}));
        (cell[mid][b] || (cell[mid][b] = [])).push(v);
        (overall[mid] || (overall[mid] = [])).push(v);
      }
    }

    const models = Object.keys(overall).sort((a, b) => mean(overall[b]) - mean(overall[a]));
    if (!models.length || !buckets.length) {
      const e = document.createElement("div");
      e.className = "empty";
      e.textContent = "Not enough data to break the benchmark down on this axis.";
      out.appendChild(e);
      return;
    }

    const bestByBucket = {};
    for (const b of buckets) {
      let bm = null, bv = -Infinity;
      for (const mid of models) {
        const xs = (cell[mid] || {})[b];
        if (!xs || !xs.length) continue;
        const m = mean(xs);
        if (m > bv) { bv = m; bm = mid; }
      }
      bestByBucket[b] = bm;
    }

    const table = document.createElement("table");
    table.className = "bd-table";
    const thead = document.createElement("thead");
    const htr = document.createElement("tr");
    const h0 = document.createElement("th"); h0.className = "txt"; h0.textContent = "Model";
    const h1 = document.createElement("th"); h1.className = "txt"; h1.textContent = "Family";
    htr.append(h0, h1);
    for (const b of buckets) {
      const th = document.createElement("th");
      th.innerHTML = b + "<span class='bd-n'>n=" + dsCount[b] + "</span>";
      htr.appendChild(th);
    }
    const hov = document.createElement("th"); hov.textContent = "Overall"; htr.appendChild(hov);
    thead.appendChild(htr);
    table.appendChild(thead);

    const tb = document.createElement("tbody");
    for (const mid of models) {
      const m = tab.models[mid] || {};
      const tr = document.createElement("tr");
      const name = document.createElement("td"); name.className = "txt model";
      name.textContent = m.display || mid;
      const fam = document.createElement("td"); fam.className = "txt cat";
      const dot = document.createElement("span");
      dot.className = "dot";
      dot.style.background = COLORS[m.category] || COLORS["Other"] || "#999";
      fam.appendChild(dot);
      fam.appendChild(document.createTextNode(m.category || "—"));
      tr.append(name, fam);
      for (const b of buckets) {
        const td = document.createElement("td");
        const xs = (cell[mid] || {})[b];
        if (xs && xs.length) {
          td.textContent = mean(xs).toFixed(3);
          if (bestByBucket[b] === mid) td.className = "best";
        } else {
          td.textContent = "—";
          td.className = "bd-na";
        }
        tr.appendChild(td);
      }
      const ov = document.createElement("td");
      ov.textContent = mean(overall[mid]).toFixed(3);
      ov.className = "bd-overall";
      tr.appendChild(ov);
      tb.appendChild(tr);
    }
    table.appendChild(tb);
    out.appendChild(table);
  }

  build();
}

"""Self-contained reports and paper exports from an evaluated leaderboard."""

import base64
import html
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from tabbench_bio.coverage import coverage_counts
from tabbench_bio.metrics import PRIMARY_CLF_METRIC, PRIMARY_REG_METRIC


def paired_comparisons(means, focus, *, lower_is_better=False):
    """Pair dataset means, with pointwise bootstrap intervals and Holm correction."""
    rows = []
    for model in means.columns:
        if model == focus:
            continue
        paired = means[[focus, model]].dropna()
        delta = np.round((paired[focus] - paired[model]).to_numpy(), 12)
        n = len(delta)
        if n >= 2:
            samples = np.random.default_rng(0).integers(n, size=(2000, n))
            low, high = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975])
        else:
            low, high = np.nan, np.nan
        p = (
            float(wilcoxon(delta, zero_method="pratt", method="approx").pvalue)
            if n >= 2 and np.any(delta)
            else (1.0 if n >= 2 else np.nan)
        )
        oriented = -delta if lower_is_better else delta
        rows.append(
            {
                "model": model,
                "datasets": n,
                "wins": int((oriented > 0).sum()),
                "ties": int((oriented == 0).sum()),
                "losses": int((oriented < 0).sum()),
                "mean_delta": float(delta.mean()) if n else np.nan,
                "ci_low": low,
                "ci_high": high,
                "p_raw": p,
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "model",
            "datasets",
            "wins",
            "ties",
            "losses",
            "mean_delta",
            "ci_low",
            "ci_high",
            "p_raw",
        ],
    )
    frame["p_holm"] = np.nan
    valid = frame["p_raw"].dropna().sort_values()
    # Keep the full planned family size even if a comparator has no paired targets.
    corrected = np.maximum.accumulate(valid.to_numpy() * (len(frame) - np.arange(len(valid))))
    frame.loc[valid.index, "p_holm"] = np.minimum(corrected, 1)
    return frame


def _table(frame, focus, *, index=False):
    rendered = frame.to_html(
        index=index,
        border=0,
        na_rep="—",
        float_format=lambda x: f"{x:.4f}",
        classes="sortable",
        escape=True,
    )
    rendered = rendered.replace(
        "<td>" + html.escape(focus) + "</td>",
        '<td class="focus">' + html.escape(focus) + "</td>",
    )
    return '<div class="scroll">' + rendered + "</div>"


def _latex(frame, caption):
    escape = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    def cell(value):
        text = (
            "--" if pd.isna(value) else (f"{value:.4f}" if isinstance(value, float) else str(value))
        )
        return re.sub(r"[\\&%$#_{}~^]", lambda match: escape[match[0]], text)

    rows = [
        " & ".join(cell(value) for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    ]
    return (
        "\\begin{table*}[t]\n\\centering\\scriptsize\n\\resizebox{\\textwidth}{!}{%\n"
        + "\\begin{tabular}{"
        + "l" * len(frame.columns)
        + "}\n\\toprule\n"
        + " & ".join(cell(column) for column in frame.columns)
        + " \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}}\n\\caption{"
        + cell(caption)
        + "}\n\\end{table*}\n"
    )


def _figure(figure, output, name):
    for extension in ("svg", "png", "pdf"):
        figure.savefig(output / (name + "." + extension), bbox_inches="tight", dpi=180)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(figure)
    return (
        '<img alt="'
        + html.escape(name.replace("_", " "))
        + '" src="data:image/svg+xml;base64,'
        + base64.b64encode(buffer.getvalue()).decode("ascii")
        + '">'
    )


def write_report(
    lb,
    ranking,
    output,
    *,
    focus,
    task,
    cell,
    config,
    status,
    reference,
    model_settings="Not recorded",
):
    """Write a portable HTML report, CSV tables, LaTeX tables and vector figures."""
    output = Path(output)
    models = ranking["model_id"].tolist()
    assert focus in models, f"Model absent from comparison: {focus}"
    labels = ranking.set_index("model_id")["Model"].to_dict()
    labels[focus] = focus
    figure = lb.plot(task)
    elo_image = _figure(figure, output, "leaderboard")
    ranking.to_csv(output / "leaderboard.csv", index=False)
    rank_columns = ["Rank", "Model", "Elo", "Elo_lo", "Elo_hi", "Score", "Avg Rank", "# Targets"]
    rank_table = ranking[rank_columns].rename(
        columns={"Elo_lo": "CI low", "Elo_hi": "CI high", "Score": "Normalized score"}
    )
    (output / "leaderboard.tex").write_text(
        _latex(rank_table, "Fold-level Elo; 95 percent dataset-bootstrap intervals. RF = 1000."),
        encoding="utf-8",
    )
    sections, all_metrics = [], []
    for selected_task, frame, metric, lower in (
        ("classification", lb._clf_metrics, PRIMARY_CLF_METRIC, False),
        ("regression", lb._reg_metrics, PRIMARY_REG_METRIC, True),
    ):
        if task not in ("overall", selected_task) or frame.empty:
            continue
        frame = frame.loc[frame["model"].isin(models)].copy()
        assert not frame.duplicated(["key", "seed", "model"]).any(), "Duplicate scored folds"
        means = (
            frame.groupby(["key", "model"])[metric].mean().unstack("model").reindex(columns=models)
        )
        field = "datasets_" + selected_task
        if field in config:
            order = {name: position for position, name in enumerate(config[field])}
            keys = sorted(
                means.index,
                key=lambda key: (
                    order[key.rsplit("_", 1)[0]] if key.rsplit("_", 1)[0] in order else len(order),
                    key,
                ),
            )
            means = means.reindex(keys)
        counts = (
            frame.groupby(["key", "model"])["seed"].nunique().unstack("model").reindex_like(means)
        )
        # Match fold identities for a paired target; equal counts alone are insufficient.
        seeds = frame.groupby(["key", "model"])["seed"].agg(frozenset)
        for key in means.index:
            if (key, focus) not in seeds.index:
                continue
            for model in models:
                if (key, model) in seeds.index:
                    assert seeds[key, model] == seeds[key, focus], (
                        f"Mismatched paired folds: {key}, {model}"
                    )
        all_metrics.append(frame)
        means.to_csv(output / f"dataset_scores_{selected_task}.csv", index_label="Dataset")
        detail = frame.groupby(["key", "model"])[metric].agg(["mean", "std", "count"]).reset_index()
        detail.to_csv(output / f"dataset_summary_{selected_task}.csv", index=False)
        pairs = paired_comparisons(means, focus, lower_is_better=lower)
        pairs.to_csv(output / f"paired_comparisons_{selected_task}.csv", index=False)
        clean = frame.loc[~frame["imputed"].eq(True)] if "imputed" in frame else frame
        clean_counts = (
            clean.groupby(["key", "model"])["seed"].nunique().unstack("model").reindex_like(means)
        )
        successful = means.where(clean_counts.eq(counts))
        sensitivity = paired_comparisons(successful, focus, lower_is_better=lower)
        sensitivity.to_csv(output / f"paired_successful_{selected_task}.csv", index=False)
        shown = [model for model in models[:5] if model != focus]
        shown = [focus] + shown
        paper = means[shown].rename(columns=labels).reset_index().rename(columns={"key": "Dataset"})
        (output / f"dataset_scores_{selected_task}.tex").write_text(
            _latex(
                paper,
                f"Mean {metric} across recorded outer folds. Fold SDs and all models are in the CSV tables.",
            ),
            encoding="utf-8",
        )
        (output / f"paired_comparisons_{selected_task}.tex").write_text(
            _latex(
                pairs,
                f"Dataset-level comparisons with {focus}. Delta = model minus baseline; pointwise 95 percent bootstrap intervals; Holm-corrected Wilcoxon tests.",
            ),
            encoding="utf-8",
        )
        values = means.rank(axis=1, ascending=lower, method="average") if lower else means
        figure, axis = plt.subplots(
            figsize=(max(7, len(models) * 0.44 + 3), max(4, len(means) * 0.28 + 1.8))
        )
        heat = axis.imshow(
            values.to_numpy(), aspect="auto", cmap="viridis_r" if lower else "viridis"
        )
        axis.set_xticks(
            range(len(models)),
            [labels[model] for model in models],
            rotation=60,
            ha="right",
            fontsize=8,
        )
        axis.set_yticks(range(len(means)), means.index, fontsize=8)
        axis.set_title("Dataset ranks (lower is better)" if lower else "Mean macro-F1", loc="left")
        figure.colorbar(heat, ax=axis, shrink=0.5)
        heatmap = _figure(figure, output, "dataset_scores_" + selected_task)
        best = next(
            (model for model in models if model != focus and means[model].notna().any()), ""
        )
        scatter = ""
        if best:
            paired = means[[focus, best]].dropna()
            if not paired.empty:
                figure, axis = plt.subplots(figsize=(5, 4))
                axis.scatter(paired[best], paired[focus], color="#d97706", s=22, alpha=0.8)
                low, high = np.nanmin(paired.to_numpy()), np.nanmax(paired.to_numpy())
                if low == high:
                    low, high = low - 0.01, high + 0.01
                axis.plot([low, high], [low, high], "--", color="#94a3b8", linewidth=1)
                axis.set_xlabel(labels[best] + " · " + metric)
                axis.set_ylabel(focus + " · " + metric)
                axis.set_title(f"{len(paired)} paired datasets", loc="left")
                scatter = _figure(figure, output, "paired_" + selected_task)
        pair_display = pairs.drop(columns="p_raw").rename(
            columns={
                "model": "Baseline",
                "datasets": "Datasets",
                "mean_delta": "Δ score",
                "p_holm": "Holm p",
            }
        )
        pair_display["Baseline"] = pair_display["Baseline"].map(labels)
        score_name = "Macro-F1 ↑" if not lower else "RMSE ↓"
        focus_mean = means[focus].mean()
        sections.append(
            f"<section><h2>{selected_task.title()} <small>{score_name} · {len(means)} targets · model mean {focus_mean:.4f}</small></h2>"
            + heatmap
            + _table(
                means.rename(columns=labels).reset_index().rename(columns={"key": "Dataset"}), focus
            )
            + '<h3>Paired comparisons</h3><p class="muted">Δ = your model − baseline. W/T/L follows metric direction.</p>'
            + _table(pair_display, focus)
            + scatter
            + "<details><summary>Successful fits only</summary>"
            + _table(sensitivity.drop(columns="p_raw"), focus)
            + "</details></section>"
        )
    metrics = pd.concat(all_metrics, ignore_index=True)
    assert not metrics.empty
    selected_status = status.loc[status["model"].isin(models) & status["key"].isin(metrics["key"])]
    assert not selected_status.duplicated(["seed", "key", "model"]).any(), (
        "Duplicate attempt records"
    )
    coverage = metrics.groupby("model").agg(Targets=("key", "nunique"), Scored=("seed", "size"))
    coverage["Imputed"] = (
        metrics.assign(imputed=metrics["imputed"].eq(True) if "imputed" in metrics else False)
        .groupby("model")["imputed"]
        .sum()
        .astype(int)
    )
    recorded = (
        coverage_counts(selected_status)
        .set_index("model_id")
        .rename(columns={"# Passed": "Passed", "# Failed": "Failed", "# Skipped": "Skipped"})
    )
    coverage = coverage.join(recorded).reindex(models)
    coverage.to_csv(output / "coverage.csv", index_label="model")
    reasons = (
        selected_status.loc[selected_status["status"].isin(["fail", "skip"])]
        .groupby(["model", "status", "reason"], dropna=False)
        .size()
        .rename("folds")
        .reset_index()
    )
    reasons.to_csv(output / "failure_reasons.csv", index=False)
    focus_row = ranking.loc[ranking["model_id"] == focus].iloc[0]
    units = metrics.loc[metrics["model"] == focus]
    cards = [
        ("Rank", f"{focus_row['Rank']} / {len(models)}"),
        ("Elo · 95% CI", f"{focus_row['Elo']} [{focus_row['Elo_lo']}, {focus_row['Elo_hi']}]"),
        ("Targets", str(units["key"].nunique())),
        ("Scored folds", str(len(units))),
    ]
    settings = {
        "Cell": cell,
        "Reference": reference,
        "Folds": config["cv_folds"],
        "Support cap": config["train_subsample"],
        "Feature cap": config["bio_max_features"],
        "Estimator": model_settings,
    }
    settings_table = pd.DataFrame(
        {"Setting": settings.keys(), "Value": [str(value) for value in settings.values()]}
    )
    methods = (
        "<details><summary>Methods</summary>"
        + _table(settings_table, focus)
        + "<p>Fold-level Bradley–Terry Elo; RF = 1000; 2,000 dataset-bootstrap samples. Score is target-normalized, not raw F1/RMSE. Paired tests use dataset means, two-sided Wilcoxon/Pratt (normal approximation), Holm correction per task. Δ intervals: 2,000 paired-dataset bootstrap samples, seed 0. Fold SDs are descriptive. Shared cohorts can make targets dependent.</p><p>Recorded failures retain benchmark imputation; successful-only comparisons remove imputed targets. Missing attempts are unknown. No matched memory/runtime comparison.</p></details>"
    )
    icon = "data:image/svg+xml;base64," + base64.b64encode(
        (Path(__file__).parent / "web/assets/favicon.svg").read_bytes()
    ).decode("ascii")
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TabBench-Bio · '
        + html.escape(focus)
        + '</title><link rel="icon" type="image/svg+xml" href="'
        + icon
        + '"><style>'
        + STYLE
        + '</style></head><body><div class="site-header"><div class="wordmark"><img src="'
        + icon
        + '" alt="" aria-hidden="true"><span>TabBench-<strong>Bio</strong></span></div></div><header><p class="eyebrow">'
        + html.escape(cell)
        + "</p><h1>"
        + html.escape(focus)
        + '</h1></header><div class="cards">'
    )
    page += (
        "".join(
            "<div><span>"
            + html.escape(label)
            + "</span><strong>"
            + html.escape(value)
            + "</strong></div>"
            for label, value in cards
        )
        + "</div>"
    )
    page += (
        "<section><h2>Ranking</h2>"
        + elo_image
        + _table(rank_table, focus)
        + "</section>"
        + "".join(sections)
    )
    page += (
        '<section><h2>Coverage</h2><p class="muted">Counts are folds. Imputed scores remain in the ranking.</p>'
        + _table(coverage.rename(index=labels).reset_index(), focus)
        + "<details><summary>Failure and skip reasons</summary>"
        + _table(reasons, focus)
        + "</details></section>"
        + methods
        + "<script>"
        + SCRIPT
        + "</script></body></html>"
    )
    (output / "report.html").write_text(page, encoding="utf-8")


STYLE = """
:root{color-scheme:light}*{box-sizing:border-box}body{font:15px/1.5 system-ui,sans-serif;color:#172033;background:#f5f7fb;max-width:1250px;margin:40px auto;padding:0 24px}h1{font-size:32px;margin:4px 0 24px}h2{font-size:21px;margin:0 0 18px}h3{font-size:17px;margin-top:30px}.eyebrow,.muted,small{color:#637087}.eyebrow{font-size:12px;letter-spacing:.08em}small{font-weight:400;font-size:13px;display:block}section,details,.cards>div{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:22px;margin:20px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.cards>div{margin:0}.cards span{display:block;color:#637087;font-size:12px}.cards strong{display:block;font-size:21px;margin-top:5px}img{max-width:100%;height:auto;display:block;margin:12px auto 24px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}td,th{text-align:right;padding:9px;border-bottom:1px solid #edf0f4}td:first-child,th:first-child{text-align:left}th{cursor:pointer;color:#637087}tbody tr:has(.focus){background:#fff5e8}.focus{font-weight:700;color:#b45309}summary{cursor:pointer;font-weight:600}details p{font-size:12px;color:#637087}.filter{font:inherit;font-size:12px;border:1px solid #d8dfe9;border-radius:5px;padding:7px;margin-bottom:10px}@media(max-width:700px){body{padding:0 12px;margin:22px auto}.cards{grid-template-columns:repeat(2,1fr)}section{padding:15px}.cards strong{font-size:17px}}@media print{body{background:white;max-width:none}.filter{display:none}section{break-inside:avoid;border:0}}
"""

STYLE += """
:root{--bg:#f4f5f1;--surface:#ffffff;--surface-strong:#ecefe8;--ink:#161a18;--muted:#616862;--line:#d9ded8;--accent:#176b52;--accent-soft:#dfeee7;--amber:#8a4d08;--amber-soft:#f3e7d7}
body{color:var(--ink);background:var(--bg)}.site-header{display:flex;justify-content:space-between;align-items:center;gap:24px;min-height:68px;margin-bottom:32px;border-bottom:1px solid var(--line)}.wordmark{display:inline-flex;align-items:center;gap:10px}.wordmark img{width:26px;height:26px;margin:0}.wordmark strong{color:var(--accent)}.eyebrow{color:var(--accent)}.muted,small,.cards span,th,details p{color:var(--muted)}section,details,.cards>div{background:var(--surface);border-color:var(--line);border-radius:9px}section{box-shadow:0 14px 40px rgba(24,38,31,.04)}.cards>div{border-top:3px solid var(--accent)}td,th{border-bottom-color:var(--line)}thead{background:var(--surface-strong)}tbody tr:has(.focus){background:var(--amber-soft)}.focus{color:var(--amber)}.filter{border-color:var(--line);background:var(--surface);color:var(--ink)}.filter:focus-visible,summary:focus-visible,th:focus-visible{outline:2px solid var(--accent);outline-offset:3px}@media(max-width:700px){.site-header{gap:12px;margin-bottom:24px}}@media print{section{box-shadow:none}}
"""

SCRIPT = """
document.querySelectorAll('table.sortable').forEach(table=>{
 const input=document.createElement('input');input.className='filter';input.placeholder='Filter rows';input.setAttribute('aria-label','Filter table rows');table.parentNode.before(input);
 input.addEventListener('input',()=>Array.from(table.tBodies[0].rows).forEach(row=>row.hidden=!row.textContent.toLowerCase().includes(input.value.toLowerCase())));
 table.querySelectorAll('thead th').forEach((th,col)=>{th.tabIndex=0;th.setAttribute('aria-sort','none');
 const sort=()=>{const asc=th.getAttribute('aria-sort')!=='ascending';table.querySelectorAll('th').forEach(h=>h.setAttribute('aria-sort','none'));th.setAttribute('aria-sort',asc?'ascending':'descending');
 const body=table.tBodies[0];Array.from(body.rows).sort((a,b)=>{const x=a.cells[col].textContent.trim(),y=b.cells[col].textContent.trim();if(x==='—'||y==='—')return (x==='—')-(y==='—');const nx=Number(x),ny=Number(y);return (Number.isNaN(nx)||Number.isNaN(ny)?x.localeCompare(y):nx-ny)*(asc?1:-1)}).forEach(row=>body.append(row));};
 th.addEventListener('click',sort);th.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();sort()}});
 });
});
"""

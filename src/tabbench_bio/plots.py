"""Leaderboard figures."""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch


def plot_elo_ranking(ranking, *, scope, highlighted=(), figsize=None):
    """Draw the ordered, already rated models and their Elo confidence intervals."""
    if ranking.empty:
        raise ValueError("No anchored Elo to plot. Include RF and at least one comparable model.")
    df = ranking.iloc[::-1]
    model_col = "Model" if "Model" in df.columns else "model_id"
    models = df[model_col].tolist()
    scores = df["Elo"].to_numpy(dtype=float)
    colors = ["#d97706" if model in highlighted else "#2563eb" for model in df["model_id"]]
    fig = Figure(figsize=figsize or (10, max(8, 0.3 * len(models) + 2)), facecolor="white")
    ax = fig.subplots()
    y = np.arange(len(models))
    ax.errorbar(
        scores,
        y,
        xerr=np.vstack([scores - df["Elo_lo"], df["Elo_hi"] - scores]),
        fmt="none",
        ecolor="#64748b",
        capsize=3,
        linewidth=1.3,
        zorder=2,
    )
    ax.scatter(scores, y, c=colors, s=60, zorder=3)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11, color="#111827")
    ax.set_xlabel("Fold-level Bradley–Terry Elo · Random Forest = 1000", fontsize=11)
    ax.set_title("TabBench-Bio", loc="left", fontsize=19, fontweight="bold", pad=22)
    ax.text(
        0,
        1.015,
        f"{scope} · 95% target-bootstrap intervals",
        transform=ax.transAxes,
        fontsize=10,
        color="#475569",
    )
    ax.axvline(1000, color="#64748b", linewidth=1, linestyle="--", zorder=1)
    ax.grid(axis="x", color="#e2e8f0")
    ax.set_axisbelow(True)
    ax.margins(y=0.12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if highlighted:
        legend = [
            Patch(color="#2563eb", label="Baselines"),
            Patch(color="#d97706", label="Your model"),
        ]
        ax.legend(handles=legend, loc="lower right")

    fig.tight_layout()
    return fig

#!/bin/bash

#SBATCH --job-name=TabBench-Bio

#SBATCH --cpus-per-task=256
#SBATCH --nodes=1
#SBATCH --partition=h100-ferranti # --partition=h100-ferranti h100-preemptable-ferranti
#SBATCH --mem=2000G

# Multi-GPU: feature_sweep.py splits the model list by device tag (configs/models/all.json)
# and runs three concurrent pools — GPU "solo" models (the memory-heavy foundation models,
# tagged "solo": true) pinned one-per-GPU so they never share VRAM; the remaining light GPU
# models packed several per GPU; and a GPU-free CPU pool for RF/LR/... so the CPU models never
# hold a GPU hostage. Tuning knobs (exported below):
#   GPU_WORKERS_PER_DEVICE  *shared* (non-solo) units packed onto each (80GB) GPU
#   GPU_SOLO_WORKERS        max concurrent solo (whole-GPU) foundation-model units (default #GPUs)
#   CPU_POOL_WORKERS        width of the CPU pool (default ~ cores/16)
#SBATCH --gres=gpu:4

#SBATCH --time=72:00:00

#SBATCH --chdir=/weka/pfeifer/ppu738/TabBench-Bio
#SBATCH --error=/weka/pfeifer/ppu738/TabBench-Bio/logs/run_grid.%j.err
#SBATCH --output=/weka/pfeifer/ppu738/TabBench-Bio/logs/run_grid.%j.out


# Single run-and-forget entry point for the whole benchmark. The 2-D feature x sample HDLSS
# grid sweep (configs/grid_sweep.json) is the *only* compute: there is no separate main
# benchmark. The grid's untruncated full-sample cell (cap_full/) is itself a complete run of
# every model on every dataset, so the static site — leaderboard tabs, the per-type
# Breakdown, the Datasets tab AND the grid Elo Leaderboard / performance surfaces — is built
# from it directly. Every model fit is therefore computed exactly once.
#
# NOTE: datasets_classification and datasets_regression in configs/grid_sweep.json drive
# which leaderboard tabs appear; the grid Elo / surface sections are classification-only.
PROJECT_ROOT=/weka/pfeifer/ppu738/TabBench-Bio
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"
source "$PROJECT_ROOT/init.sh"

# Pack two *shared* (non-solo) units per 80GB GPU; solo-tagged foundation models still get a
# whole GPU each.
export GPU_WORKERS_PER_DEVICE=2

# Run the heavy solo foundation models (MITRA, ...) at most 2 at a time instead of one-per-GPU,
# to keep concurrent host-RAM/VRAM peaks down; the freed GPUs still take shared-tier units.
export GPU_SOLO_WORKERS=2

# Cap the CPU pool at 8 workers (default would be cores/16 = 16 here). The last run was
# host-RAM OOM-killed with all 45 SIGKILLs in this pool, so hold its concurrent dataset copies down.
export CPU_POOL_WORKERS=8

GRID_CONFIG=configs/grid_sweep.json

echo "############################################################"
echo "STEP 1/2 — feature x sample grid sweep ($GRID_CONFIG)"
echo "############################################################"
python scripts/feature_sweep.py --grid-config "$GRID_CONFIG"

# The site draws the grid heatmaps / learning curves interactively (docs/js/charts.js from the
# persisted CSVs below); no static images are produced.

# Persist the grid aggregates into the git-tracked docs/data (results/ is gitignored), so the
# site step below folds them into the Elo Leaderboard / performance surfaces.
if [ -f results/feature_sweep/sweep_summary.csv ]; then
    cp results/feature_sweep/sweep_summary.csv docs/data/feature_grid_summary.csv
    cp results/feature_sweep/sweep_metrics.csv docs/data/feature_grid_metrics.csv
    echo "Persisted grid aggregates -> docs/data/feature_grid_{summary,metrics}.csv"
else
    echo "ERROR: results/feature_sweep/sweep_summary.csv missing." >&2
    exit 1
fi

echo "############################################################"
echo "STEP 2/2 — static site from the grid's untruncated full-sample cell"
echo "############################################################"
# The leaderboard / Breakdown / Datasets tabs are built from the full-feature, full-sample
# grid cell (no separate benchmark run). Prefer the uncapped cell (cap_full/); fall back to
# the widest numeric cap if the grid has no 'full' cap. Read straight from grid_sweep.json so
# this stays in sync with the configured caps.
LEADERBOARD_CELL=$(python - <<'PY'
import json, os
gc = json.load(open("configs/grid_sweep.json"))
def is_full(c): return c is None or (isinstance(c, str) and c.lower() in ("full", "none", "null", ""))
caps = gc["caps"]
name = "cap_full" if any(is_full(c) for c in caps) else "cap_%d" % max(int(c) for c in caps)
print(os.path.join(gc.get("output", "results/feature_sweep"), name))
PY
)
echo "leaderboard cell: $LEADERBOARD_CELL"
tabbench-bio site --results-dir "$LEADERBOARD_CELL" --out docs --config "$LEADERBOARD_CELL/config.json"

echo "############################################################"
echo "ALL DONE."
echo "  site        : docs/index.html  (leaderboard from $LEADERBOARD_CELL)"
echo "  grid CSVs   : docs/data/feature_grid_{summary,metrics}.csv"
echo "  grid figs   : docs/assets/grid_{heatmap,curves}_*.png"
echo "############################################################"

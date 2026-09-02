#!/bin/bash
#SBATCH --job-name=Bio-finalize
#SBATCH --partition=cpu-ferranti
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/weka/pfeifer/ppu738/TabBench-Bio
#SBATCH --error=/weka/pfeifer/ppu738/TabBench-Bio/logs/grid_finalize.%j.err
#SBATCH --output=/weka/pfeifer/ppu738/TabBench-Bio/logs/grid_finalize.%j.out

set -euo pipefail
PROJECT_ROOT=/weka/pfeifer/ppu738/TabBench-Bio
cd "$PROJECT_ROOT"
mkdir -p logs
source "$PROJECT_ROOT/init.sh"

GRID_CONFIG=configs/grid_sweep_all.json
RESULTS_ROOT=results/feature_sweep_all_v4

# Materialise only dataset-level design skips that the iterator independently reproduces.
# Missing model runs and non-null targets remain absent and fail the coverage gate below.
python scripts/materialize_null_dataset_skips.py --results-root "$RESULTS_ROOT"

# Recompute every cell's metrics after all retry jobs have stopped.  Existing rows are
# reused, while newly landed predictions are appended before the completeness gate below.
cell_configs=("$RESULTS_ROOT"/cap_*/config.json)
if [[ ! -f "${cell_configs[0]}" ]]; then
  echo "No grid-cell configs found under $RESULTS_ROOT" >&2
  exit 2
fi
for config in "${cell_configs[@]}"; do
  python -m tabbench_bio.cli run --config "$config" --step metrics
done

# Refuse to aggregate if any configured cell or scheduled unit is absent.  This reuses the
# benchmark's publication gate without invoking its legacy site renderer.
python - "$GRID_CONFIG" "$RESULTS_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

from tabbench_bio.config import load_config
from tabbench_bio.coverage import assert_complete, load_status
from tabbench_bio.seeds import get_seeds
from tabbench_bio.site import _expected_keys

grid_config_path = Path(sys.argv[1])
results_root = Path(sys.argv[2])
grid_config = json.loads(grid_config_path.read_text(encoding="utf-8"))
cell_configs = sorted(results_root.glob("cap_*/config.json"))
expected_cells = len(grid_config["caps"]) * len(grid_config["samples"])
assert len(cell_configs) == expected_cells, (
    f"Expected {expected_cells} grid cells, found {len(cell_configs)} under {results_root}."
)

for config_path in cell_configs:
    config = load_config(str(config_path))
    cell_dir = config_path.parent
    frames = []
    for filename in ("classification_metrics.csv", "regression_metrics.csv"):
        path = cell_dir / "metrics" / filename
        if path.is_file():
            frames.append(pd.read_csv(path))
    assert frames, f"No metrics found for {cell_dir.name}."
    assert_complete(
        pd.concat(frames, ignore_index=True),
        load_status(str(cell_dir)),
        keys=_expected_keys(config),
        models=config["models"],
        seeds=get_seeds(config),
    )
    print(f"complete: {cell_dir.name}", flush=True)
PY

python scripts/feature_sweep.py --grid-config "$GRID_CONFIG" --skip-run
echo "Final aggregation complete. Build paper and site artifacts from the frozen result tree."

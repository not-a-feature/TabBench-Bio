#!/bin/bash
#SBATCH --job-name=Bio-grid-full
#SBATCH --partition=h100-ferranti
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --gres=gpu:h100:5
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/weka/pfeifer/ppu738/TabBench-Bio
#SBATCH --error=/weka/pfeifer/ppu738/TabBench-Bio/logs/grid_full.%j.err
#SBATCH --output=/weka/pfeifer/ppu738/TabBench-Bio/logs/grid_full.%j.out

set -euo pipefail
PROJECT_ROOT=/weka/pfeifer/ppu738/TabBench-Bio
cd "$PROJECT_ROOT"
mkdir -p logs
source "$PROJECT_ROOT/init.sh"

# The full cell contains the one-hour AutoGluon reference. Five GPUs let its five CV folds
# advance concurrently, keeping the worst-case reference runtime near 37 rather than 185 hours.
export GPU_WORKERS_PER_DEVICE=2
export GPU_SOLO_WORKERS=5
export CPU_POOL_WORKERS=8

# See run_grid_array.sbatch: the OOMs here are fragmentation, not a size ceiling.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/feature_sweep.py \
  --grid-config configs/grid_sweep_all.json \
  --shard-count "${GRID_SHARDS:?GRID_SHARDS must be exported by the submitter}" \
  --shard-index "$((GRID_SHARDS - 1))" \
  --skip-prepare

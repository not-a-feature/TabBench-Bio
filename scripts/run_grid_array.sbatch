#!/bin/bash
#SBATCH --job-name=Bio-grid
#SBATCH --partition=h100-ferranti
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --chdir=/weka/pfeifer/ppu738/TabBench-Bio
#SBATCH --error=/weka/pfeifer/ppu738/TabBench-Bio/logs/grid_array.%A_%a.err
#SBATCH --output=/weka/pfeifer/ppu738/TabBench-Bio/logs/grid_array.%A_%a.out

set -euo pipefail
PROJECT_ROOT=/weka/pfeifer/ppu738/TabBench-Bio
cd "$PROJECT_ROOT"
mkdir -p logs
source "$PROJECT_ROOT/init.sh"

# One GPU per independent cell shard. Four CPU workers overlap classical models with GPU fits.
export GPU_WORKERS_PER_DEVICE=2
export GPU_SOLO_WORKERS=1
export CPU_POOL_WORKERS=4

# The foundation-model OOMs on this grid land while the card is well under capacity (24 GiB
# allocations failing at ~40 GiB used), i.e. they are fragmentation, not a size ceiling.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/feature_sweep.py \
  --grid-config configs/grid_sweep_all.json \
  --shard-count "${GRID_SHARDS:?GRID_SHARDS must be exported by the submitter}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --skip-prepare

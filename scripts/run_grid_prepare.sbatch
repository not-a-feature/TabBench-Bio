#!/bin/bash
#SBATCH --job-name=Bio-prepare
#SBATCH --partition=cpu-ferranti
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --chdir=/weka/pfeifer/ppu738/TabBench-Bio
#SBATCH --error=/weka/pfeifer/ppu738/TabBench-Bio/logs/grid_prepare.%j.err
#SBATCH --output=/weka/pfeifer/ppu738/TabBench-Bio/logs/grid_prepare.%j.out

set -euo pipefail
PROJECT_ROOT=/weka/pfeifer/ppu738/TabBench-Bio
cd "$PROJECT_ROOT"
mkdir -p logs
source "$PROJECT_ROOT/init.sh"

# Protein-embedding tables are deliberately not committed or re-hosted. A separate,
# reproducible GPU curation job materializes them from the checksum-pinned upstream data.
for local_dataset in \
  deeploc2_fungi_esm2_t33.parquet \
  peer_beta_lactamase_esm2_t33.parquet; do
  path="$PROJECT_ROOT/src/tabbench_bio/bio/data/local/$local_dataset"
  if [[ ! -s "$path" ]]; then
    echo "Missing required protein embedding table: $path" >&2
    echo "Submit scripts/generate_protein_embeddings.sbatch before this preparation job." >&2
    exit 2
  fi
done

# Reuse the raw, dataset-id-addressed cache, but isolate processed split caches from the
# currently running legacy sweep. This prevents concurrent index/config writes.
mkdir -p .cache/grid_all
if [ ! -e .cache/grid_all/bio ]; then
  ln -s ../bio .cache/grid_all/bio
fi
python scripts/feature_sweep.py \
  --grid-config configs/grid_sweep_all.json \
  --prepare-only

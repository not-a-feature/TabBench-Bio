#!/bin/bash
# Generate evo2 / ntv2 / dnabert2 embeddings for the BRCA VEP eval set and write the
# three TabBench-Bio "local" dataset CSVs (embedding + y=func.class). Single GPU; models
# run sequentially cheap->expensive (dnabert2, ntv2, evo2) so bugs surface fast.
#
# Submit:  sbatch scripts/generate_brca_embeddings.sbatch
# Smoke:   sbatch scripts/generate_brca_embeddings.sbatch --models dnabert2 --limit 8 \
#                 --out-dir /weka/pfeifer/ppu738/TabBench-Bio/scratch/brca_smoke

#SBATCH --job-name=brca-embed

#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --partition=h100-ferranti
#SBATCH --mem=200G

#SBATCH --gres=gpu:1

#SBATCH --time=24:00:00

#SBATCH --chdir=/weka/pfeifer/ppu738/TabBench-Bio
#SBATCH --error=/weka/pfeifer/ppu738/TabBench-Bio/logs/brca_embed.%j.err
#SBATCH --output=/weka/pfeifer/ppu738/TabBench-Bio/logs/brca_embed.%j.out


PROJECT_ROOT=/weka/pfeifer/ppu738/TabBench-Bio
mkdir -p "$PROJECT_ROOT/logs"

# Activate the proven DNA-foundation-models env (evo2 + transformers 4.37.2 + flash-attn +
# the DNABERT-2 Triton deps) — the same env that produced the GABBA embeddings.
# (Sourced before `set -u`: ~/.bashrc -> /etc/bashrc references unset vars.)
source /weka/pfeifer/ppu738/DNA_Foundation_Models_Inversion_Attack_Benchmark/init.sh

set -eo pipefail
cd "$PROJECT_ROOT"
echo "[brca-embed] python: $(which python)"
nvidia-smi -L || true

# Forward any extra args (e.g. --limit, --models, --out-dir) straight to the script.
python "$PROJECT_ROOT/scripts/generate_brca_embeddings.py" "$@"

echo "[brca-embed] finished."

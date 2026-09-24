#!/usr/bin/env bash
#SBATCH --job-name=TabBench-model
#SBATCH --account=l40s
#SBATCH --partition=L40Sday
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=90G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/model.%j.out
#SBATCH --error=logs/model.%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the TabBench-Bio checkout}"
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TABPFN_NO_BROWSER=1
export TABPFN_MODEL_CACHE_DIR="$PWD/.cache/tabpfn"
if [[ -n ${TABPFN_CREDENTIAL_FILE:-} ]]; then
    source "$TABPFN_CREDENTIAL_FILE"
fi
export TMPDIR="/scratch/${SLURM_JOB_ID:?}"
mkdir -p "$TMPDIR" "$TABPFN_MODEL_CACHE_DIR"
# The core CLI selects .venvs/<profile> from the model registry.
# Match the benchmark: 18 allocated CPUs, 32 model/library threads.
srun --cpu-bind=threads .venv/bin/tabbench-bio "${@:?Pass a model key, optionally --full-grid}"

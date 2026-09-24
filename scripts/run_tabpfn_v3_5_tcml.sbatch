#!/usr/bin/env bash
#SBATCH --job-name=TabBench-V35
#SBATCH --account=l40s
#SBATCH --partition=L40Sday
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=90G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/tabpfn-v35.%A_%a.out
#SBATCH --error=logs/tabpfn-v35.%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the TabBench-Bio checkout}"
model_python="$PWD/.venvs/tabpfn35/bin/python"
[[ -x "$model_python" ]] || { echo "Install environments/tabpfn35.txt in .venvs/tabpfn35 first." >&2; exit 1; }
export VIRTUAL_ENV="$PWD/.venvs/tabpfn35"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export TABPFN_NO_BROWSER=1
export TABPFN_MODEL_CACHE_DIR="$PWD/.cache/tabpfn"
export TABBENCH_MODEL_CPUS=32
export TABBENCH_BIO_RESOURCE_TIER=l40s
export TABBENCH_RESOURCE_PROFILE=l40s-18c-90g-32threads-v1
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
export TMPDIR="/scratch/${SLURM_JOB_ID:?}"
mkdir -p "$TMPDIR" "$TABPFN_MODEL_CACHE_DIR"
if [[ -n ${TABPFN_CREDENTIAL_FILE:-} ]]; then
    source "$TABPFN_CREDENTIAL_FILE"
fi

# Smoke-test both model versions before writing any benchmark results.
srun --cpu-bind=threads "$model_python" scripts/smoke_tabpfn_versions.py
shard_args=()
if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
    # Use a contiguous, zero-based array to give each worker disjoint grid cells.
    [[ $SLURM_ARRAY_TASK_MIN == 0 ]]
    [[ $SLURM_ARRAY_TASK_MAX == $((SLURM_ARRAY_TASK_COUNT - 1)) ]]
    shard_args=(--shard-count "$SLURM_ARRAY_TASK_COUNT" --shard-index "$SLURM_ARRAY_TASK_ID")
fi
srun --cpu-bind=threads "$model_python" scripts/feature_sweep.py \
    --grid-config configs/grid_tabpfn_v3_5.json \
    --cache-dir "${TABBENCH_CACHE_DIR:-.cache}" \
    --include-device gpu "${shard_args[@]}" "$@"

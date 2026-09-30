#!/usr/bin/env bash
# Usage: sbatch --array=0-7%8 scripts/run_grid_tcml.sbatch PROFILE GRID_CONFIG
# Prepare the cell configs and shared data cache before submitting the array.
#SBATCH --job-name=TabBench-grid
#SBATCH --account=l40s
#SBATCH --partition=L40Sday
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=90G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/grid.%A_%a.out
#SBATCH --error=logs/grid.%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the TabBench-Bio checkout}"
profile="${1:?Pass an environment profile}"
grid="${2:?Pass a prepared grid config}"
shift 2
model_python="$(.venv/bin/python -c 'import sys; from tabbench_bio.model_run import model_python; print(model_python(sys.argv[1]))' "$profile")"
export VIRTUAL_ENV="$PWD/.venvs/$profile"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export TABBENCH_MODEL_CPUS="${SLURM_CPUS_PER_TASK:?Missing CPU allocation}"
export OMP_NUM_THREADS="$TABBENCH_MODEL_CPUS"
export OPENBLAS_NUM_THREADS="$TABBENCH_MODEL_CPUS" MKL_NUM_THREADS="$TABBENCH_MODEL_CPUS"
export TMPDIR="/scratch/${SLURM_JOB_ID:?}"
mkdir -p "$TMPDIR"
[[ ${SLURM_ARRAY_TASK_MIN:?Submit a contiguous, zero-based array} == 0 ]]
[[ $SLURM_ARRAY_TASK_MAX == $((SLURM_ARRAY_TASK_COUNT - 1)) ]]
srun --cpu-bind=threads "$model_python" scripts/feature_sweep.py \
    --grid-config "$grid" --include-device gpu --skip-prepare --skip-metrics \
    --shard-count "$SLURM_ARRAY_TASK_COUNT" --shard-index "$SLURM_ARRAY_TASK_ID" "$@"

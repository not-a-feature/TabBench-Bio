#!/bin/bash
set -euo pipefail

# Resume is idempotent: passing prediction/status files are skipped and only missing/failed
# units are fitted again. By default retry every regular cell and the dedicated full cell.
GRID_SHARDS=${GRID_SHARDS:-28}
GRID_CONCURRENCY=${GRID_CONCURRENCY:-16}
ARRAY_SPEC=${ARRAY_SPEC:-0-$((GRID_SHARDS - 2))}
SUBMIT_FULL=${SUBMIT_FULL:-1}

if ! [[ "$GRID_SHARDS" =~ ^[0-9]+$ ]] || (( GRID_SHARDS < 2 )); then
  echo "GRID_SHARDS must be an integer of at least 2" >&2
  exit 2
fi
if ! [[ "$GRID_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "GRID_CONCURRENCY must be a positive integer" >&2
  exit 2
fi
if [[ "$SUBMIT_FULL" != 0 && "$SUBMIT_FULL" != 1 ]]; then
  echo "SUBMIT_FULL must be 0 or 1" >&2
  exit 2
fi

if [[ "$ARRAY_SPEC" != "none" ]]; then
  array_job=$(sbatch --parsable \
    --array="${ARRAY_SPEC}%${GRID_CONCURRENCY}" \
    --export="ALL,GRID_SHARDS=${GRID_SHARDS}" \
    run_grid_array.sbatch)
  printf 'array=%s (indices %s)\n' "$array_job" "$ARRAY_SPEC"
fi

if [[ "$SUBMIT_FULL" == 1 ]]; then
  full_job=$(sbatch --parsable \
    --export="ALL,GRID_SHARDS=${GRID_SHARDS}" \
    run_grid_full.sbatch)
  printf 'full=%s\n' "$full_job"
fi

printf '%s\n' "Finalize only after every retry is complete: scripts/submit_grid_finalize.sh"

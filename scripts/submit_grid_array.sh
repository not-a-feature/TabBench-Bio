#!/bin/bash
set -euo pipefail

# The default 28 shards map one-to-one to the 4 feature caps x 7 sample budgets. The reference
# cell is submitted separately at nice=0 so it becomes available early; the remaining array is
# deliberately less favoured while still running concurrently.
GRID_SHARDS=${GRID_SHARDS:-28}
GRID_CONCURRENCY=${GRID_CONCURRENCY:-16}
REGULAR_NICE=${REGULAR_NICE:-1000}
REFERENCE_CAP=${REFERENCE_CAP:-10000}
REFERENCE_SAMPLES=${REFERENCE_SAMPLES:-100}
GRID_CONFIG=${GRID_CONFIG:-configs/grid_sweep_all.json}
if ! [[ "$GRID_SHARDS" =~ ^[0-9]+$ ]] || (( GRID_SHARDS < 2 )); then
  echo "GRID_SHARDS must be an integer of at least 2" >&2
  exit 2
fi
if ! [[ "$GRID_CONCURRENCY" =~ ^[0-9]+$ ]] || (( GRID_CONCURRENCY < 2 )); then
  echo "GRID_CONCURRENCY must be an integer of at least 2" >&2
  exit 2
fi
if ! [[ "$REGULAR_NICE" =~ ^[0-9]+$ ]]; then
  echo "REGULAR_NICE must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$REFERENCE_CAP" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "$REFERENCE_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "REFERENCE_CAP and REFERENCE_SAMPLES must be positive integers" >&2
  exit 2
fi

reference_shard=$(python -c '
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
target = (int(sys.argv[2]), int(sys.argv[3]))
expected_shards = int(sys.argv[4])
cells = [(cap, sample) for cap in config["caps"] for sample in config["samples"]]
assert len(cells) == expected_shards, (
    f"GRID_SHARDS={expected_shards}, but {sys.argv[1]} defines {len(cells)} cells."
)
assert cells[-1] == (None, None), "The dedicated final shard must remain the full/full cell."
assert target in cells, f"Reference cell {target} is absent from {sys.argv[1]}."
print(cells.index(target))
' "$GRID_CONFIG" "$REFERENCE_CAP" "$REFERENCE_SAMPLES" "$GRID_SHARDS")
if (( reference_shard >= GRID_SHARDS - 1 )); then
  echo "Reference shard must be distinct from the dedicated full/full shard" >&2
  exit 2
fi

regular_indices=()
for ((index = 0; index < GRID_SHARDS - 1; index++)); do
  if (( index != reference_shard )); then
    regular_indices+=("$index")
  fi
done
if (( ${#regular_indices[@]} == 0 )); then
  echo "No regular grid shards remain after excluding the reference and full cells" >&2
  exit 2
fi
regular_array=$(IFS=,; echo "${regular_indices[*]}")
regular_concurrency=$((GRID_CONCURRENCY - 1))

prep_job=$(sbatch --parsable run_grid_prepare.sbatch)
reference_job=$(sbatch --parsable \
  --dependency="afterok:${prep_job}" \
  --array="${reference_shard}" \
  --nice=0 \
  --job-name=Bio-reference \
  --export="ALL,GRID_SHARDS=${GRID_SHARDS}" \
  run_grid_array.sbatch)
array_job=$(sbatch --parsable \
  --dependency="afterok:${prep_job}" \
  --array="${regular_array}%${regular_concurrency}" \
  --nice="${REGULAR_NICE}" \
  --export="ALL,GRID_SHARDS=${GRID_SHARDS}" \
  run_grid_array.sbatch)
# The last shard is the full/full cell and is submitted separately with five GPUs so its five
# one-hour-reference CV folds run concurrently.
full_job=$(sbatch --parsable \
  --dependency="afterok:${prep_job}" \
  --export="ALL,GRID_SHARDS=${GRID_SHARDS}" \
  run_grid_full.sbatch)
finalize_job=$(sbatch --parsable \
  --dependency="afterany:${reference_job}:${array_job}:${full_job}" \
  run_grid_finalize.sbatch)

printf 'prepare=%s\nreference=%s (shard %s: cap=%s, samples=%s)\narray=%s\nfull=%s\nfinalize=%s\n' \
  "$prep_job" "$reference_job" "$reference_shard" "$REFERENCE_CAP" "$REFERENCE_SAMPLES" \
  "$array_job" "$full_job" "$finalize_job"

#!/bin/bash
set -euo pipefail

# run_grid_finalize.sbatch validates that every configured cell has metrics before publishing.
job=$(sbatch --parsable run_grid_finalize.sbatch)
printf 'finalize=%s\n' "$job"

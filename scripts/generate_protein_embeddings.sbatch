#!/bin/bash
# Materialize the two checksum-pinned protein embedding datasets on one H100.
#
# Submit: sbatch scripts/generate_protein_embeddings.sbatch
# Audit:  python scripts/generate_protein_embeddings.py --audit-only

#SBATCH --job-name=protein-embed
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --chdir=/weka/pfeifer/ppu738/TabBench-Bio
#SBATCH --error=/weka/pfeifer/ppu738/TabBench-Bio/logs/protein_embed.%j.err
#SBATCH --output=/weka/pfeifer/ppu738/TabBench-Bio/logs/protein_embed.%j.out

set -euo pipefail
PROJECT_ROOT=/weka/pfeifer/ppu738/TabBench-Bio
cd "$PROJECT_ROOT"
mkdir -p logs
source "$PROJECT_ROOT/init.sh"

python - <<'PY'
import importlib.util

required = ("lmdb", "pyarrow", "torch", "transformers")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing embedding dependencies: "
        + ", ".join(missing)
        + ". Reinstall with `pip install -e '.[bio,models]'`."
    )
PY

nvidia-smi -L
python scripts/generate_protein_embeddings.py --device cuda --batch-size 4 --force "$@"

echo "[protein-embed] finished."

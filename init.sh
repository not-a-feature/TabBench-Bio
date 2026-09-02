#!/bin/bash
# Environment bootstrap for TabBench-Bio (mirrors the layout of other cluster projects).
# Creates/activates a uv-managed virtualenv and installs the package with the AutoGluon
# fork that lifts the 500-feature cap on tabular foundation models (required for omics).
USER_HOME=/weka/pfeifer/ppu738
BASE_DIR_LOCAL=$USER_HOME/TabBench-Bio
VENV_DIR=$BASE_DIR_LOCAL/.venv
PYTHON_VERSION=3.11   # AutoGluon 1.5 supports 3.9-3.12; 3.11 is the safe choice

# Keep all model/dataset caches inside the project (avoids filling the home quota).
# XDG_CACHE_HOME redirects the generic ~/.cache (OpenML, uv, ...) into the repo;
# the remaining tools use their own cache env var and are pinned explicitly.
export XDG_CACHE_HOME=$BASE_DIR_LOCAL/.cache
export TABBENCH_BIO_CACHE=$BASE_DIR_LOCAL/.cache/bio
export HF_HOME=$BASE_DIR_LOCAL/.cache/huggingface
export TABPFN_MODEL_CACHE_DIR=$BASE_DIR_LOCAL/.cache/tabpfn
export KAGGLEHUB_CACHE=$BASE_DIR_LOCAL/.cache/kagglehub

TABPFN_ENV_FILE=$USER_HOME/.config/tabbench-bio/tabpfn.env
if [[ ! -r "$TABPFN_ENV_FILE" ]]; then
    echo "Missing TabPFN credential file: $TABPFN_ENV_FILE" >&2
    exit 2
fi
source "$TABPFN_ENV_FILE"
: "${TABPFN_TOKEN:?TABPFN_TOKEN is empty in $TABPFN_ENV_FILE}"

# Let the CUDA allocator grow/shrink segments instead of pre-reserving fixed blocks: cuts the
# fragmentation-driven OOM a long-lived GPU worker hits fitting many models back-to-back.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$BASE_DIR_LOCAL"

### uv bootstrap
# uv ships in the local miniconda; fall back to the standalone installer if absent.
export PATH="$USER_HOME/miniconda3/bin:$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create + populate the venv on first run, otherwise just activate it.
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating uv venv '$VENV_DIR' (python $PYTHON_VERSION)..."
    uv venv --python $PYTHON_VERSION "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    # 1) AutoGluon fork FIRST (git installs that satisfy the autogluon.* extras below)
    echo "Installing AutoGluon fork (lifts the 500-feature cap)..."
    uv pip install -r requirements-autogluon-fork.txt

    # 2) TabBench-Bio with everything (autogluon extras are already satisfied by the fork)
    echo "Installing tabbench-bio[full] (editable)..."
    uv pip install -e ".[full]"
else
    echo "Activating existing uv venv '$VENV_DIR'..."
    source "$VENV_DIR/bin/activate"
fi

# The venv is populated once at creation, so source-only deps added to pyproject.toml later
# (tabfm ships from git, not PyPI) can be missing from an existing venv. Guard-install it when
# absent — a no-op once present. Runs on the compute node where the venv interpreter exists.
python -c "import tabfm" 2>/dev/null || {
    echo "Installing missing tabfm backend (git source)..."
    uv pip install "tabfm[pytorch] @ git+https://github.com/google-research/tabfm"
}

# Molecular TDC datasets use RDKit. Existing long-lived cluster environments predate
# this loader, so install the optional bio dependency once when it is missing.
python -c "import rdkit" 2>/dev/null || {
    echo "Installing missing RDKit molecular backend..."
    uv pip install "rdkit>=2024.3"
}
### End uv

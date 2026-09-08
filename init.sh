#!/usr/bin/env bash
# Source this file from a shell to install and activate the full benchmark environment.
TABBENCH_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TABBENCH_VENV_DIR="${TABBENCH_VENV_DIR:-$TABBENCH_PROJECT_DIR/.venv}"

if [[ ! -d "$TABBENCH_VENV_DIR" ]]; then
    python3.11 -m venv "$TABBENCH_VENV_DIR" || return
fi
source "$TABBENCH_VENV_DIR/bin/activate"
python -m pip install -r "$TABBENCH_PROJECT_DIR/requirements-autogluon-fork.txt" || return
python -m pip install -e "$TABBENCH_PROJECT_DIR[full]" || return

export TABBENCH_BIO_CACHE="${TABBENCH_BIO_CACHE:-$TABBENCH_PROJECT_DIR/.cache/bio}"
export HF_HOME="${HF_HOME:-$TABBENCH_PROJECT_DIR/.cache/huggingface}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$TABBENCH_PROJECT_DIR/.cache/tabpfn}"

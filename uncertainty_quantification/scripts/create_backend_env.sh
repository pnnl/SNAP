#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/create_backend_env.sh mace [venv_path]
  scripts/create_backend_env.sh uma [venv_path]

Examples:
  scripts/create_backend_env.sh mace
  scripts/create_backend_env.sh uma
  PYTHON=/opt/homebrew/opt/python@3.12/bin/python3.12 scripts/create_backend_env.sh mace

This creates a backend-specific virtual environment and installs uq-mlip with
the matching backend extra. MACE and UMA currently require separate
environments because their e3nn dependencies conflict.
USAGE
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

backend="$1"
case "$backend" in
  mace|uma) ;;
  *)
    echo "Unsupported backend: $backend" >&2
    usage
    exit 2
    ;;
esac

venv_path="${2:-.venv-${backend}}"
python_bin="${PYTHON:-python3}"

"$python_bin" - <<'PY'
import sys

if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
    raise SystemExit(
        f"Python {sys.version.split()[0]} is unsupported. Use Python >=3.11,<3.14."
    )
PY

"$python_bin" -m venv "$venv_path"
# shellcheck disable=SC1091
source "$venv_path/bin/activate"

python -m pip install -U pip
python -m pip install -e ".[dev,${backend}]"

if [[ "$(uname -s)" == "Darwin" ]]; then
  if [[ ! -e /opt/homebrew/opt/libomp/lib/libomp.dylib && ! -e /usr/local/opt/libomp/lib/libomp.dylib ]]; then
    cat <<'NOTE'

macOS note:
  XGBoost requires the OpenMP runtime for training.
  Install it with:

    brew install libomp

NOTE
  fi
fi

if [[ "$backend" == "uma" ]]; then
  cat <<'NOTE'

UMA note:
  FairChem writes model/cache files at import time. On locked-down systems,
  set FAIRCHEM_CACHE_DIR to a writable directory before running UMA commands:

    export FAIRCHEM_CACHE_DIR=/path/to/writable/fairchem-cache

NOTE
fi

cat <<DONE

Created $backend environment at: $venv_path

Activate it with:
  source "$venv_path/bin/activate"

Verify:
  uq-mlip --help
DONE

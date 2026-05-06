#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_backend_hello_world.sh mace [venv_path] [outdir]
  scripts/run_backend_hello_world.sh uma [venv_path] [outdir]

This runs extract -> train -> predict -> plot on a tiny XYZ file using the
selected backend. It may download model weights the first time the backend is
used.
USAGE
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 2
fi

backend="$1"
venv_path="${2:-.venv-${backend}}"
outdir="${3:-examples/hello_world/outputs-${backend}}"
train_sample="examples/test_data/water_train.xyz"
run_sample="examples/test_data/water_run.xyz"

case "$backend" in
  mace|uma) ;;
  *)
    echo "Unsupported backend: $backend" >&2
    usage
    exit 2
    ;;
esac

if [[ ! -f "$venv_path/bin/activate" ]]; then
  echo "Virtual environment not found: $venv_path" >&2
  echo "Create it first with: scripts/create_backend_env.sh $backend $venv_path" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$venv_path/bin/activate"

mkdir -p "$outdir"
export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-$outdir/fairchem-cache}"

if [[ "$backend" == "mace" ]]; then
  uq-mlip extract \
    --backend mace \
    --sample "$train_sample" \
    --savedir "$outdir/embeddings" \
    --device cpu \
    --model-size small \
    --index ":"
  uq-mlip extract \
    --backend mace \
    --sample "$run_sample" \
    --savedir "$outdir/embeddings" \
    --device cpu \
    --model-size small \
    --index ":"
else
  uq-mlip extract \
    --backend uma \
    --sample "$train_sample" \
    --savedir "$outdir/embeddings" \
    --device cpu \
    --model-size uma-s-1p1 \
    --head omat \
    --index ":"
  uq-mlip extract \
    --backend uma \
    --sample "$run_sample" \
    --savedir "$outdir/embeddings" \
    --device cpu \
    --model-size uma-s-1p1 \
    --head omat \
    --index ":"
fi

train_embedding_file="$outdir/embeddings/embedding_info_water_train.npz"
run_embedding_file="$outdir/embeddings/embedding_info_water_run.npz"

uq-mlip train \
  --embeddings "$train_embedding_file" \
  --savedir "$outdir/uq-model" \
  --estimators 25

uq-mlip predict \
  --embeddings "$run_embedding_file" \
  --model-dir "$outdir/uq-model" \
  --savedir "$outdir/results"

python examples/hello_world/plot_uq_profile.py \
  --csv "$outdir/results/UQ_embedding_info_water_run.csv.gz" \
  --output "$outdir/uq_profile.svg" \
  --title "uq-mlip $backend hello world"

echo "Wrote $outdir/uq_profile.svg"

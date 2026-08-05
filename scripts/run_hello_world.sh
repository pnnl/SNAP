#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_hello_world.sh mace [venv_path] [outdir] [train_sample] [run_sample]
  scripts/run_hello_world.sh uma [venv_path] [outdir] [train_sample] [run_sample]
  scripts/run_hello_world.sh chgnet [venv_path] [outdir] [train_sample] [run_sample]

This runs extract -> train -> predict -> plot on a tiny XYZ file using the
selected backend. It may download model weights the first time the backend is
used.
USAGE
}

if [[ $# -lt 1 || $# -gt 5 ]]; then
  usage
  exit 2
fi

backend="$1"
venv_path="${2:-.venv-${backend}}"
outdir="${3:-examples/hello_world/outputs-${backend}}"
train_sample="${4:-examples/test_data/water_train.xyz}"
run_sample="${5:-examples/test_data/water_run.xyz}"
train_stem="$(basename "$train_sample")"
train_stem="${train_stem%.*}"
run_stem="$(basename "$run_sample")"
run_stem="${run_stem%.*}"

case "$backend" in
  mace|uma|chgnet) ;;
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
elif [[ "$backend" == "chgnet" ]]; then
  uq-mlip extract \
    --backend chgnet \
    --sample "$train_sample" \
    --savedir "$outdir/embeddings" \
    --device cpu \
    --index ":"
  uq-mlip extract \
    --backend chgnet \
    --sample "$run_sample" \
    --savedir "$outdir/embeddings" \
    --device cpu \
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

train_embedding_file="$outdir/embeddings/embedding_info_${train_stem}.npz"
run_embedding_file="$outdir/embeddings/embedding_info_${run_stem}.npz"

uq-mlip train \
  --embeddings "$train_embedding_file" \
  --savedir "$outdir/uq-model" \
  --estimators 25

uq-mlip predict \
  --embeddings "$run_embedding_file" \
  --model-dir "$outdir/uq-model" \
  --savedir "$outdir/results"

python examples/hello_world/plot_uq_profile.py \
  --csv "$outdir/results/UQ_embedding_info_${run_stem}.csv.gz" \
  --output "$outdir/uq_profile.svg" \
  --title "uq-mlip $backend hello world"

echo "Wrote $outdir/uq_profile.svg"

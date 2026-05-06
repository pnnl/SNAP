# Validation plan before merge

This branch should not be merged until the workflow has been checked at three
levels.

## 1. Core package workflow

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pytest
uq-mlip --help
python examples/hello_world/train_run_visualize.py
```

Expected artifacts:

- `examples/hello_world/outputs/embeddings/embedding_info_train.npz`
- `examples/hello_world/outputs/embeddings/embedding_info_run.npz`
- `examples/hello_world/outputs/results/UQ_synthetic_run.csv.gz`
- `examples/hello_world/outputs/uq_profile.svg`

The backend workflows use the small static test dataset in `examples/test_data/`:

- `water_train.xyz`
- `water_run.xyz`

On macOS, run `brew install libomp` before training with XGBoost.

## 2. MACE backend environment

```bash
scripts/create_backend_env.sh mace
scripts/run_backend_hello_world.sh mace
```

This verifies:

- MACE dependency installation in its own environment
- backend embedding extraction
- UQ model training
- prediction CSV generation
- UQ profile PNG generation

## 3. UMA backend environment

```bash
scripts/create_backend_env.sh uma
scripts/run_backend_hello_world.sh uma
```

This verifies the same path for UMA. UMA may require model-download access and
any authentication expected by FairChem/Hugging Face in the user's environment.
If `~/.cache` is not writable, set `FAIRCHEM_CACHE_DIR` to a writable path.

## Why separate environments?

MACE and UMA currently depend on incompatible `e3nn` versions. Keep them in
separate environments until those upstream constraints become compatible.

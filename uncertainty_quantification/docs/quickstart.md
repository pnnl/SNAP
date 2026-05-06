# uq-mlip quickstart

`uq-mlip` adds per-atom uncertainty quantification to machine learning
interatomic potentials. The UQ model is not universal: train it for the MLIP
and dataset you plan to use.

## Install

MACE and UMA are supported out of the box as default model backends. Install the
core package first:

```bash
pip install uq-mlip
```

Then install the backend dependency stack you plan to use:

```bash
pip install uq-mlip[mace]
pip install uq-mlip[uma]
```

For a local checkout, use `pip install -e .` and then install the matching
editable extra, for example `pip install -e ".[mace]"`.

MACE and UMA currently depend on incompatible `e3nn` versions, so use separate
environments if you need to run both backends.

The setup scripts automate this:

```bash
scripts/create_backend_env.sh mace
scripts/create_backend_env.sh uma
```

On macOS, install OpenMP before training the GBM model with XGBoost:

```bash
brew install libomp
```

For UMA on systems where `~/.cache` is not writable, set:

```bash
export FAIRCHEM_CACHE_DIR=/path/to/writable/fairchem-cache
```

## Train a UQ model

Start with structures from the validation set, or data representative of where
the MLIP will be used.

```bash
uq-mlip extract \
  --backend mace \
  --sample validation.xyz \
  --savedir embeddings/

uq-mlip train \
  --embeddings embeddings/embedding_info_validation.npz \
  --savedir uq-model/ \
  --lower-alpha 0.05 \
  --upper-alpha 0.95 \
  --estimators 1000
```

UMA uses the same workflow:

```bash
uq-mlip extract \
  --backend uma \
  --sample validation.xyz \
  --savedir embeddings/ \
  --model-size uma-m-1p1 \
  --head omat
```

## Hello world

The synthetic hello world runs without downloading MACE or UMA model weights:

```bash
python examples/hello_world/train_run_visualize.py
```

It writes:

- `examples/hello_world/outputs/results/UQ_synthetic_run.csv.gz`
- `examples/hello_world/outputs/uq_profile.svg`

To verify a real backend end to end, use a backend-specific environment:

```bash
scripts/run_backend_hello_world.sh mace
scripts/run_backend_hello_world.sh uma
```

These backend workflows use the small test dataset in `examples/test_data/`.

## Use UQ in existing code

Decorator-style calculator:

```python
from uq_mlip import UQCalculator

atoms.calc = UQCalculator(
    base_calculator=mace_calc,
    uq_model="uq-model/",
    backend="mace",
    model="medium-0b",
)
```

Convenience helper:

```python
from uq_mlip import with_uq

atoms.calc = with_uq(
    mace_calc,
    uq_model="uq-model/",
    backend="mace",
    model="medium-0b",
)
```

Normal energy and force calls continue to work. Per-atom UQ values are available
as `atoms.arrays["uq"]`, `atoms.arrays["uq_lower"]`, and
`atoms.arrays["uq_upper"]` after a calculation.

## Add a new MLIP backend

Implement an extractor with two methods:

```python
class MyExtractor:
    def extract_atoms(self, atoms):
        ...

    def extract_file(self, sample, savedir, index=":"):
        ...
```

Both methods should return or save the shared schema:

- `node_feats`: per-atom embedding matrix
- `node_energies`: per-atom energy targets for training
- `node_type`: atomic numbers
- `num_atoms`: atoms per configuration

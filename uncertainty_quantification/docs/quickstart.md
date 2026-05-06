# uq-mlip quickstart

`uq-mlip` adds per-atom uncertainty quantification to machine learning
interatomic potentials. The UQ model is not universal: train it for the MLIP
and dataset you plan to use.

## Install

MACE and UMA are supported out of the box as flagship backends:

```bash
pip install uq-mlip
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

# Change summary

## What changed

- Added installable package scaffolding through `pyproject.toml`.
- Added a new `uq_mlip` package without changing the existing research scripts.
- Added a shared embedding schema for per-atom MLIP embeddings.
- Added a quantile GBM `UQModel` API for training, loading, and prediction.
- Added out-of-box backend adapters for MACE and UMA.
- Added `UQCalculator` and `with_uq()` for minimal-change integration in ASE workflows.
- Added a `uq-mlip` CLI with `extract`, `train`, and `predict` commands.
- Added a quickstart document with the intended user workflow.
- Added a small import smoke test.

## Design notes

- The UQ model is dataset and MLIP specific. Users should train it on validation
  or representative configurations before using it in production simulations.
- MACE and UMA are first-class supported backends in the default package.
- Existing files such as `gbm.py`, `run-gbm.py`, `run_embeddings_mace.py`,
  `run_embeddings_uma.py`, and `train-gbm.py` were intentionally left unchanged.
- The new package fixes the integration surface by adding an API wrapper rather
  than changing the existing scripts in place.

## Primary usage

```bash
uq-mlip extract --backend mace --sample validation.xyz --savedir embeddings/
uq-mlip train --embeddings embeddings/embedding_info_validation.npz --savedir uq-model/
uq-mlip predict --embeddings embeddings/embedding_info_validation.npz --model-dir uq-model/ --savedir results/
```

```python
from uq_mlip import with_uq

atoms.calc = with_uq(mace_calc, uq_model="uq-model/", backend="mace", model="medium-0b")
```

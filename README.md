# uq-mlip: Per-atom Uncertainty Quantification for MLIPs

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://github.com/pnnl/SNAP/blob/master/pyproject.toml)
[![Research](https://img.shields.io/badge/papers-npj%20Computational%20Materials-4b32c3)](#research-background)
[![License](https://img.shields.io/badge/license-BSD--2--Clause-blue)](https://github.com/pnnl/SNAP/blob/master/LICENSE)

![uq-mlip banner: uncertainty quantification for neural network potentials in chemistry](https://raw.githubusercontent.com/pnnl/SNAP/master/docs/assets/banner-uq-mlip.png)

This repository contains code for training and evaluating per-atom uncertainty
quantification for machine learning interatomic potentials (MLIPs). The
recommended interface is the installable `uq-mlip` package, which provides a
unified command-line and Python API for the same extract, train, and predict
workflow available in the standalone scripts.

## New to UQ for NNPs?

> New to uncertainty quantification for neural network potentials? Start with
> the [UQ for NNPs primer](https://github.com/pnnl/SNAP/blob/master/docs/UQ_for_NNPs_Primer.pdf).

The primer gives a beginner-friendly overview of what uncertainty
quantification means for neural network potentials, why per-atom uncertainty is
useful, and how the workflow in this repository fits into simulation practice.

## Research background

`uq-mlip` implements uncertainty quantification workflows developed through two
peer-reviewed studies in *npj Computational Materials*:

- [Uncertainty quantification for neural network potential foundation models](https://doi.org/10.1038/s41524-025-01572-y)
  introduces the UQ approach for neural network potential foundation models.
- [Assessing universal MLIP robustness with per-atom uncertainty for simulations of solid-liquid interfaces](https://doi.org/10.1038/s41524-026-02051-8)
  demonstrates per-atom uncertainty as a practical diagnostic for MLIP
  robustness in challenging interfacial simulations.

## Why per-atom UQ?

MLIP predictions can look stable even when local atomic environments are
outside the model's reliable domain. `uq-mlip` adds per-atom uncertainty
estimates so simulations can be inspected, filtered, or monitored at the level
where failures often begin.

## Quick start

`uq-mlip` makes per-atom uncertainty quantification easy to add to MLIP
workflows. The UQ model is specific to the MLIP and dataset, so users should
train a UQ model on validation or representative configurations before using it
in simulations.

MACE and UMA are supported out of the box as default model backends:

```
git clone https://github.com/pnnl/SNAP.git
cd SNAP
git checkout master
pip install -e .
```

To include a backend in the same environment:

```
pip install -e ".[mace]"  # for MACE extraction
pip install -e ".[uma]"   # for UMA extraction
```

MACE and UMA currently depend on incompatible `e3nn` versions, so use separate
environments if you need to exercise both dependency stacks. After the first
PyPI release, this section will be updated to use `pip install uq-mlip`.

To automate backend-specific setup:

```
scripts/create_backend_env.sh mace
scripts/create_backend_env.sh uma
```

On macOS, XGBoost may also require the OpenMP runtime:

```
brew install libomp
```

For UMA on systems where `~/.cache` is not writable:

```
export FAIRCHEM_CACHE_DIR=/path/to/writable/fairchem-cache
```

Train a UQ model:

```
uq-mlip extract \
  --backend mace \
  --sample validation.xyz \
  --savedir embeddings/

uq-mlip train \
  --embeddings embeddings/embedding_info_validation.npz \
  --savedir uq-model/
```

Run the local hello world example to train a small UQ model, predict a synthetic
trajectory, and generate a UQ profile visualization:

```
python examples/hello_world/train_run_visualize.py
```

To make a tiny complete-frame fixture from a large XYZ/extXYZ trajectory:

```
python scripts/slice_xyz.py /path/to/large.xyz examples/test_data/my_slice.xyz --frames 8
```

To smoke-test a real backend in its own environment:

```
scripts/run_hello_world.sh mace
scripts/run_hello_world.sh uma
```

You can pass custom train/run slices to the backend smoke test without copying a
large trajectory into the repository:

```
scripts/run_hello_world.sh mace .venv-mace outputs/mace-small \
  examples/test_data/aimd_pbe_train.xyz examples/test_data/aimd_pbe_run.xyz
```

Use it in existing code with the decorator-style calculator:

```python
from uq_mlip import UQCalculator

atoms.calc = UQCalculator(
    base_calculator=mace_calc,
    uq_model="uq-model/",
    backend="mace",
    model="medium-0b",
)
```

Or use the convenience helper:

```python
from uq_mlip import with_uq

atoms.calc = with_uq(
    mace_calc,
    uq_model="uq-model/",
    backend="mace",
    model="medium-0b",
)
```

Energy and force calls continue to behave like the original calculator. After a
calculation, per-atom uncertainty is available as `atoms.arrays["uq"]`,
`atoms.arrays["uq_lower"]`, and `atoms.arrays["uq_upper"]`.

To verify a local checkout:

```
pip install -e ".[dev]"
python -m pytest
uq-mlip --help
python examples/hello_world/train_run_visualize.py
```

## Standalone script interface

The commands below expose the same workflow through the original script-level
entry points. They remain useful for direct inspection, debugging, and
reproducing the paper-era workflow.

### Extract embeddings and per-atom energies

To train the GBM model, first extract per-atom embeddings and per-atom energies
from a trained MLIP. The following commands provide methods to extract this
information for MACE and UMA. If using a finetuned checkpoint, the
`--checkpoint` flag can be used to specify the path to the checkpoint file. The
sample should be in a format readable by ASE and contain configurations *from
the validation set* used to train or finetune the MLIP.

```
python run_embeddings_mace.py \
  --sample data/example.xyz \
  --savedir data/embeddings_mace \
  --model-size medium-0b \
  --index ":"
```

```
python run_embeddings_uma.py \
  --sample data/example.xyz \
  --savedir data/embeddings_uma \
  --model-size uma-s-1p1 \
  --head 'omat' \
  --index ":"
```


### Train GBM on per-atom embeddings and energies

Once the embeddings and energies have been extracted, the GBM model can be
trained using the following command.

```
python train-gbm.py --embeddings data/embeddings_mace/embedding_info_example.npz \
  --savedir data/gbm_mace \
  --upper-alpha 0.95 \
  --lower-alpha 0.05 \
  --estimators 1000
```


### Compute per-atom uncertainties using the trained GBM model

To compute per-atom uncertainties for a trajectory produced using the MLIP, the
per-atom embeddings must be extracted in the same way as described above.

```
python run_embeddings_mace.py \
  --sample data/md_run.xyz \
  --savedir data/embeddings_mace \
  --model-size medium-0b \
  --index ":"
```

Then, the following command can be used to compute per-atom uncertainties using
the trained GBM model.

```
python run-gbm.py --embeddings 'data/embeddings/embedding_info_md_run.npz' --savedir 'results/gbm_mace'
```

## Citation
If you use this model or code in your research, please cite the following papers:

```bibtex
@article{Bilbrey2025,
  author = {Bilbrey, Jenna A. and Firoz, Jesun S. and Lee, Mal-Soon and Choudhury, Sutanay},
  title = {Uncertainty quantification for neural network potential foundation models},
  journal = {npj Computational Materials},
  year = {2025},
  volume = {11},
  number = {1},
  pages = {109},
  doi = {10.1038/s41524-025-01572-y},
  url = {https://doi.org/10.1038/s41524-025-01572-y}
}

@article{Bilbrey2026,
  author = {Bilbrey, Jenna A. and Firoz, Jesun S. and Allec, Sarah I. and Sprueill, Henry W. and von Rueden, Alexander D. and Jackson, Benjamin A. and Raugei, Simone and Lee, Mal-Soon and Choudhury, Sutanay},
  title = {Assessing universal MLIP robustness with per-atom uncertainty for simulations of solid-liquid interfaces},
  journal = {npj Computational Materials},
  year = {2026},
  doi = {10.1038/s41524-026-02051-8},
  url = {https://doi.org/10.1038/s41524-026-02051-8}
}
```

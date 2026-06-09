---
name: uq-mlip-md-workflow
description: "Train a UQ model for a pretrained MLIP using uq-mlip, run an MD simulation decorated with per-atom uncertainty, and output predicted quantiles. Use when: training uncertainty quantification model on MLIP validation set, wrapping MACE or UMA calculator with UQ, running ASE molecular dynamics with per-atom UQ, extracting quantile predictions from MD trajectory, monitoring UQ profile during simulation."
argument-hint: "<backend: mace|uma> <validation.xyz> <md_input.xyz>"
---

# UQ-MLIP MD Workflow

Train a per-atom uncertainty quantification (UQ) model on a pretrained MLIP validation set, run an MD simulation using that MLIP, and output predicted quantiles for every frame.

## When to Use

- You have a pretrained MACE or UMA MLIP and a validation dataset
- You want to attach per-atom UQ to an existing ASE calculator
- You need to monitor model reliability during MD simulations
- You want to output quantile uncertainty estimates (e.g., 5th–95th percentile) frame-by-frame

---

## Prerequisites

Install `uq-mlip` with the appropriate backend. MACE and UMA require **separate environments** due to incompatible `e3nn` versions.

```bash
# MACE backend
pip install -e ".[mace]"

# UMA backend
pip install -e ".[uma]"
# If ~/.cache is not writable:
export FAIRCHEM_CACHE_DIR=/path/to/writable/fairchem-cache
```

You need:
- A **validation set** in any ASE-readable format (`.xyz`, extended XYZ, etc.) — configurations from the MLIP's validation split
- An **MD input configuration** (ASE-readable) and MD parameters (timestep, temperature, ensemble, steps)
- The pretrained MLIP checkpoint or model identifier

---

## Step 1: Extract Embeddings from the Validation Set

Extract per-atom embeddings and per-atom energies from the validation set using the MLIP backbone. The output is a compressed `.npz` file used for UQ model training.

### CLI

```bash
# MACE
uq-mlip extract \
  --backend mace \
  --sample validation.xyz \
  --savedir embeddings/ \
  --model-size medium-0b \
  --checkpoint /path/to/checkpoint.pt \   # omit for foundation model defaults
  --device cuda \
  --index ":"

# UMA
uq-mlip extract \
  --backend uma \
  --sample validation.xyz \
  --savedir embeddings/ \
  --model-size uma-m-1p1 \
  --head omat \
  --device cuda \
  --index ":"
```

Output: `embeddings/embedding_info_<validation_stem>.npz`

### Python

```python
from uq_mlip.backends import get_extractor

extractor = get_extractor("mace", model="medium-0b", device="cuda")
# extractor = get_extractor("uma", model="uma-m-1p1", head="omat", device="cuda")

output_path = extractor.extract_file(
    "validation.xyz",
    savedir="embeddings/",
    index=":",
)
print(output_path)
```

---

## Step 2: Train the UQ Model

Train a quantile gradient-boosted machine (GBM) on the extracted embeddings. The model learns to predict lower/upper quantile bounds for per-atom energies. **Train on the validation set, not the training set**, so the model characterizes in-distribution uncertainty.

### CLI

```bash
uq-mlip train \
  --embeddings embeddings/embedding_info_validation.npz \
  --savedir uq-model/ \
  --lower-alpha 0.05 \
  --upper-alpha 0.95 \
  --estimators 100 \
  --device cpu
```

Output: `uq-model/GBMRegressor_0.05-0.95.pkl`

### Python

```python
from uq_mlip.model import UQModel

model = UQModel(
    savedir="uq-model/",
    lower_alpha=0.05,
    upper_alpha=0.95,
    n_estimators=100,
)
model.fit(embeddings)          # embeddings is an EmbeddingData object
# or from file:
model = UQModel.train_from_file("embeddings/embedding_info_validation.npz", "uq-model/")
```

**Tuning guidance:**
- `lower_alpha` / `upper_alpha`: define the quantile interval (default 5th–95th percentile). Widen for conservative bounds.
- `n_estimators`: more rounds → better fit but slower training. 100–500 is typical.
- Increase `n_estimators` if the UQ profile on a held-out validation subset is noisy.

---

## Step 3: Run MD with the UQ-Decorated Calculator

Wrap the base MLIP ASE calculator with `UQCalculator` (or `with_uq`). Energy and forces continue to come from the base calculator; per-atom UQ is appended to `atoms.arrays` on every step.

```python
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from uq_mlip import with_uq

# --- build your base MLIP calculator ---
# MACE example:
from mace.calculators import mace_mp
base_calc = mace_mp(model="medium-0b", default_dtype="float32", device="cuda")

# UMA example:
# from fairchem.core import pretrained_mlip
# base_calc = pretrained_mlip.get_predict_unit("uma-m-1p1", device="cuda")

# --- wrap with UQ ---
uq_calc = with_uq(
    base_calc,
    uq_model="uq-model/",   # directory containing the .pkl file
    backend="mace",          # "mace" or "uma"
    model="medium-0b",       # passed to the extractor; must match base_calc
    lower_alpha=0.05,
    upper_alpha=0.95,
    device="cuda",
)

# --- load initial structure ---
atoms = read("md_input.xyz")
atoms.calc = uq_calc

# --- initialize velocities ---
temperature_K = 300
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

# --- set up MD ---
timestep_fs = 1.0
dyn = Langevin(
    atoms,
    timestep=timestep_fs * units.fs,
    temperature_K=temperature_K,
    friction=0.01 / units.fs,
)

# --- trajectory and UQ logging ---
traj_file = "md_trajectory.xyz"
uq_rows = []

def log_step():
    frame = atoms.copy()
    frame.arrays["uq"] = atoms.arrays["uq"].copy()
    frame.arrays["uq_lower"] = atoms.arrays["uq_lower"].copy()
    frame.arrays["uq_upper"] = atoms.arrays["uq_upper"].copy()
    write(traj_file, frame, append=True)
    uq_rows.append({
        "step": dyn.nsteps,
        "mean_uq": float(atoms.arrays["uq"].mean()),
        "max_uq":  float(atoms.arrays["uq"].max()),
    })

dyn.attach(log_step, interval=1)
dyn.run(500)   # number of MD steps
```

---

## Step 4: Output Predicted Quantiles

### From `atoms.arrays` (per-step, per-atom)

After each calculation, three arrays are available on `atoms`:

| Array | Description |
|---|---|
| `atoms.arrays["uq"]` | Per-atom uncertainty — half the quantile interval `(upper − lower) / 2` |
| `atoms.arrays["uq_lower"]` | Lower quantile prediction (e.g., 5th percentile) |
| `atoms.arrays["uq_upper"]` | Upper quantile prediction (e.g., 95th percentile) |

### From a saved trajectory (batch)

If the MD was run without live logging, or you want to re-score a trajectory:

```bash
# 1. Extract embeddings from the trajectory
uq-mlip extract \
  --backend mace \
  --sample md_trajectory.xyz \
  --savedir embeddings/ \
  --device cuda

# 2. Predict quantiles
uq-mlip predict \
  --embeddings embeddings/embedding_info_md_trajectory.npz \
  --model-dir uq-model/ \
  --savedir results/ \
  --lower-alpha 0.05 \
  --upper-alpha 0.95
```

Output: `results/UQ_md_trajectory.csv.gz`

Columns: `sample_idx`, `atom_idx`, `element`, `uq_lower`, `uq_upper`, `uq`

### UQ profile from CSV

```python
import pandas as pd

df = pd.read_csv("results/UQ_md_trajectory.csv.gz")
profile = (
    df.groupby("sample_idx")
    .agg(mean_uq=("uq", "mean"), max_uq=("uq", "max"))
    .reset_index()
)
print(profile)
```

### Python batch prediction

```python
from uq_mlip.data import load_embeddings
from uq_mlip.model import UQModel

embeddings = load_embeddings("embeddings/embedding_info_md_trajectory.npz")
model = UQModel.from_dir("uq-model/", lower_alpha=0.05, upper_alpha=0.95)
predictions = model.predict_embeddings(embeddings)
# predictions["lower"], predictions["upper"], predictions["uncertainty"]
```

---

## Decision Checklist

- [ ] Validation XYZ contains configurations **from the MLIP's actual validation split**, not the training set
- [ ] Backend (`mace` or `uma`) matches the base calculator — use separate conda/venv environments
- [ ] `model` / `model-size` in the extractor exactly matches the checkpoint used for `base_calc`
- [ ] `lower_alpha` / `upper_alpha` are **identical** at extract, train, and predict steps
- [ ] MD trajectory is written with per-atom arrays preserved (use `write(..., append=True)` with `frame.arrays` copied)

## Interpreting Results

- **High `uq`** (large interval) at a frame → the MLIP is operating outside its reliable domain; consider flagging or re-running with DFT
- **Sudden spike** in `max_uq` → a local atomic environment became unusual; check for structural artifacts
- **All `uq` near zero** → the trajectory stays well within the training distribution

## References

- [UQCalculator / with_uq API](../../../src/uq_mlip/api.py)
- [UQModel training and prediction](../../../src/uq_mlip/model.py)
- [Embedding data schema](../../../src/uq_mlip/data.py)
- [MACE backend extractor](../../../src/uq_mlip/backends/mace.py)
- [UMA backend extractor](../../../src/uq_mlip/backends/uma.py)
- [Hello world example](../../../examples/hello_world/train_run_visualize.py)
- [CLI reference](../../../src/uq_mlip/cli.py)

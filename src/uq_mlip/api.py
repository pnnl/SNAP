"""Drop-in UQ wrappers for existing MLIP code."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from uq_mlip.backends import get_extractor
from uq_mlip.model import UQModel


class UQCalculator(Calculator):
    """ASE calculator decorator that adds per-atom UQ to an existing calculator."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        base_calculator,
        uq_model: Union[UQModel, str, Path],
        backend: str = "mace",
        extractor=None,
        lower_alpha: float = 0.05,
        upper_alpha: float = 0.95,
        **backend_kwargs: Any,
    ):
        super().__init__()
        self.base_calculator = base_calculator
        self.uq_model = (
            UQModel.from_dir(
                uq_model,
                lower_alpha=lower_alpha,
                upper_alpha=upper_alpha,
            )
            if isinstance(uq_model, (str, Path))
            else uq_model
        )
        self.extractor = extractor or get_extractor(backend, **backend_kwargs)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        energy = self.base_calculator.get_potential_energy(atoms)
        forces = self.base_calculator.get_forces(atoms)
        bundle = self.extractor.extract_atoms(atoms)
        predictions = self.uq_model.predict_embeddings(bundle)

        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["uq_lower"] = predictions["lower"]
        self.results["uq_upper"] = predictions["upper"]
        self.results["uq"] = predictions["uncertainty"]

        atoms.arrays["uq_lower"] = np.asarray(predictions["lower"])
        atoms.arrays["uq_upper"] = np.asarray(predictions["upper"])
        atoms.arrays["uq"] = np.asarray(predictions["uncertainty"])


def with_uq(
    base_calculator,
    uq_model: Union[UQModel, str, Path],
    backend: str = "mace",
    extractor=None,
    **kwargs: Any,
) -> UQCalculator:
    """Wrap an existing ASE calculator with per-atom UQ."""

    return UQCalculator(
        base_calculator=base_calculator,
        uq_model=uq_model,
        backend=backend,
        extractor=extractor,
        **kwargs,
    )

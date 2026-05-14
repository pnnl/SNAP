"""Shared embedding data schema used by all UQ backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass
class EmbeddingData:
    """Per-atom embedding bundle consumed by the UQ model."""

    node_feats: np.ndarray
    node_type: np.ndarray
    num_atoms: np.ndarray
    node_energies: Optional[np.ndarray] = None

    def validate(self, require_energies: bool = False) -> None:
        if self.node_feats.ndim != 2:
            raise ValueError("node_feats must be a 2D array of shape (n_atoms, n_features).")

        n_atoms = self.node_feats.shape[0]
        if len(self.node_type) != n_atoms:
            raise ValueError("node_type length must match node_feats rows.")

        if int(np.sum(self.num_atoms)) != n_atoms:
            raise ValueError("sum(num_atoms) must match node_feats rows.")

        if require_energies and self.node_energies is None:
            raise ValueError("node_energies are required for UQ model training.")

        if self.node_energies is not None and len(self.node_energies) != n_atoms:
            raise ValueError("node_energies length must match node_feats rows.")


def load_embeddings(path: Union[str, Path]) -> EmbeddingData:
    """Load an embedding bundle from the npz schema used by uq-mlip."""

    data = np.load(path)
    node_energies = data["node_energies"] if "node_energies" in data.files else None
    bundle = EmbeddingData(
        node_feats=data["node_feats"],
        node_energies=node_energies,
        node_type=data["node_type"],
        num_atoms=data["num_atoms"],
    )
    bundle.validate(require_energies=False)
    return bundle


def save_embeddings(bundle: EmbeddingData, path: Union[str, Path]) -> Path:
    """Save an embedding bundle to compressed npz."""

    bundle.validate(require_energies=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "node_feats": bundle.node_feats,
        "node_type": bundle.node_type,
        "num_atoms": bundle.num_atoms,
    }
    if bundle.node_energies is not None:
        payload["node_energies"] = bundle.node_energies

    np.savez_compressed(path, **payload)
    return path

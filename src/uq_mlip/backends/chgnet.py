"""CHGNet embedding extraction support."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from uq_mlip.data import EmbeddingData, save_embeddings


class CHGNetExtractor:
    """Extract per-atom embeddings and energies from CHGNet models.

    CHGNet exposes per-site energies and the per-atom feature vector produced
    just before the readout layer, which map directly onto the ``node_energies``
    and ``node_feats`` fields consumed by the UQ model.
    """

    def __init__(
        self,
        model: str = "0.3.0",
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 16,
        vacuum: float = 15.0,
    ):
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self.vacuum = vacuum
        self._model = None
        self._converter = None

    def _load_model(self):
        if self._model is None:
            from chgnet.graph import CrystalGraphConverter
            from chgnet.model.model import CHGNet

            if self.checkpoint is not None:
                model = CHGNet.from_file(self.checkpoint)
                if self.device is not None:
                    model = model.to(self.device)
            else:
                model = CHGNet.load(model_name=self.model, use_device=self.device)
            model.eval()
            self._model = model
            self._converter = CrystalGraphConverter()
        return self._model

    def _atoms_to_graph(self, atoms):
        from pymatgen.io.ase import AseAtomsAdaptor

        # CHGNet is a crystal model and requires a periodic cell. Wrap
        # non-periodic inputs (e.g. isolated molecules) in a vacuum box so the
        # structure can be converted to a graph.
        if not bool(np.all(atoms.get_pbc())) or atoms.cell.rank < 3:
            atoms = atoms.copy()
            atoms.center(vacuum=self.vacuum)
            atoms.pbc = True

        structure = AseAtomsAdaptor.get_structure(atoms)
        return self._converter(structure)

    def extract(self, atoms_list: Iterable) -> EmbeddingData:
        model = self._load_model()
        atoms_list = list(atoms_list)

        graphs = [self._atoms_to_graph(atoms) for atoms in atoms_list]
        predictions = model.predict_graph(
            graphs,
            task="e",
            return_site_energies=True,
            return_atom_feas=True,
            batch_size=self.batch_size,
        )
        if isinstance(predictions, dict):
            predictions = [predictions]

        node_feats = []
        node_energies = []
        node_type = []
        num_atoms = []
        for atoms, prediction in zip(atoms_list, predictions):
            node_feats.append(np.asarray(prediction["atom_fea"]))
            node_energies.append(np.asarray(prediction["site_energies"]).reshape(-1))
            node_type.append(np.asarray(atoms.get_atomic_numbers()))
            num_atoms.append(len(atoms))

        return EmbeddingData(
            node_feats=np.vstack(node_feats),
            node_energies=np.concatenate(node_energies),
            node_type=np.concatenate(node_type),
            num_atoms=np.asarray(num_atoms),
        )

    def extract_atoms(self, atoms) -> EmbeddingData:
        return self.extract([atoms])

    def extract_file(
        self,
        sample: Union[str, Path],
        savedir: Union[str, Path],
        index: str = ":",
    ) -> Path:
        from ase.io import read

        sample = Path(sample)
        savedir = Path(savedir)
        atoms_list = read(sample, index=index)
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        bundle = self.extract(atoms_list)
        return save_embeddings(bundle, savedir / f"embedding_info_{sample.stem}.npz")

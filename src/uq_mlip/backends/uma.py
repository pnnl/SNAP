"""UMA embedding extraction support."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from uq_mlip.data import EmbeddingData, save_embeddings


class UMAExtractor:
    """Extract per-atom embeddings and energies from UMA models."""

    def __init__(
        self,
        model: str = "uma-m-1p1",
        checkpoint: Optional[str] = None,
        head: str = "omat",
        device: str = "cuda",
        batch_size: int = 4,
    ):
        self.model = checkpoint or model
        self.head = head
        self.device = device
        self.batch_size = batch_size
        self._predictor = None

    def _load_predictor(self):
        if self._predictor is None:
            from fairchem.core import pretrained_mlip

            predictor = pretrained_mlip.get_predict_unit(self.model, device=self.device)
            predictor.model.eval()
            for param in predictor.model.parameters():
                param.requires_grad = False
            self._predictor = predictor
        return self._predictor

    def extract(self, atoms_list: Iterable) -> EmbeddingData:
        from fairchem.core.datasets import data_list_collater
        from fairchem.core.datasets.atomic_data import AtomicData

        atoms_list = list(atoms_list)
        predictor = self._load_predictor()
        all_atom_refs = np.asarray(predictor.atom_refs[self.head])

        node_type = np.asarray([z for atoms in atoms_list for z in atoms.get_atomic_numbers()])
        num_atoms = np.asarray([len(atoms) for atoms in atoms_list])
        atom_refs = all_atom_refs[node_type]

        to_graph = partial(
            AtomicData.from_ase,
            task_name=self.head,
            r_edges=True,
            r_data_keys=["spin", "charge"],
            max_neigh=predictor.model.module.backbone.max_neighbors,
            radius=predictor.model.module.backbone.cutoff,
        )
        configs = [to_graph(atoms) for atoms in atoms_list]

        descriptors = []
        node_energies = []
        start = 0
        for stop in range(self.batch_size, len(configs) + self.batch_size, self.batch_size):
            batch = data_list_collater(configs[start:stop], otf_graph=True)
            predictor.predict(batch)

            node_features = predictor.model.module.backbone.forward(batch)["node_embedding"]
            energy_input = node_features.narrow(1, 0, 1).squeeze(1)
            energy_output = predictor.model.module.output_heads.energyandforcehead.head.energy_block(
                energy_input
            )
            descriptors.extend(node_features[:, 0].detach().cpu())
            node_energies.append(energy_output.view(-1).detach().cpu())
            start = stop

        return EmbeddingData(
            node_feats=np.vstack(descriptors),
            node_energies=np.concatenate(node_energies, axis=0) + atom_refs,
            node_type=node_type,
            num_atoms=num_atoms,
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

"""MACE embedding extraction support."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from uq_mlip.data import EmbeddingData, save_embeddings


class MACEExtractor:
    """Extract per-atom embeddings and energies from MACE models."""

    def __init__(
        self,
        model: str = "medium-0b",
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        default_type: str = "float32",
    ):
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        self.default_type = default_type
        self._head = None

    def _load_head(self):
        if self._head is None:
            import torch
            from mace import data
            from mace.calculators import mace_mp
            from mace.tools import torch_geometric, utils

            class InteractionHead:
                def __init__(self, model, checkpoint, device, default_type):
                    self.device = device
                    self.charges_key = "Qs"
                    self.calc = mace_mp(model=model, default_dtype=default_type, device=device)
                    if checkpoint is not None:
                        state = torch.load(checkpoint, map_location=device)["state_dict"]
                        self.calc.model.load_state_dict({key: value for key, value in state.items()})
                    self.r_max = self.calc.models[0].r_max.item()
                    self.z_table = utils.AtomicNumberTable(
                        [int(z) for z in self.calc.models[0].atomic_numbers]
                    )

                def _atoms_to_batch(self, atoms):
                    try:
                        config = data.config_from_atoms(atoms)
                    except TypeError:
                        config = data.config_from_atoms(atoms, charges_key=self.charges_key)

                    loader = torch_geometric.dataloader.DataLoader(
                        dataset=[
                            data.AtomicData.from_config(
                                config,
                                z_table=self.z_table,
                                cutoff=self.r_max,
                            )
                        ],
                        batch_size=1,
                        shuffle=False,
                        drop_last=False,
                    )
                    return next(iter(loader)).to(self.device)

                def forward(self, atoms):
                    batch = self._atoms_to_batch(atoms)
                    return self.calc.models[0](batch)

            self._head = InteractionHead(
                model=self.model,
                checkpoint=self.checkpoint,
                device=self.device,
                default_type=self.default_type,
            )
        return self._head

    def extract_atoms(self, atoms) -> EmbeddingData:
        head = self._load_head()
        out = head.forward(atoms)
        return EmbeddingData(
            node_feats=out["node_feats"].detach().cpu().numpy(),
            node_energies=out["node_energy"].detach().cpu().numpy(),
            node_type=np.asarray(atoms.get_atomic_numbers()),
            num_atoms=np.asarray([len(atoms)]),
        )

    def extract(self, atoms_list: Iterable) -> EmbeddingData:
        node_feats = []
        node_energies = []
        node_type = []
        num_atoms = []

        for atoms in atoms_list:
            bundle = self.extract_atoms(atoms)
            node_feats.append(bundle.node_feats)
            node_energies.append(bundle.node_energies)
            node_type.append(bundle.node_type)
            num_atoms.append(len(atoms))

        return EmbeddingData(
            node_feats=np.vstack(node_feats),
            node_energies=np.concatenate(node_energies),
            node_type=np.concatenate(node_type),
            num_atoms=np.asarray(num_atoms),
        )

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

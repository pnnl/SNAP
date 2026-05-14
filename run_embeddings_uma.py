'''
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.
 
Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.
 
                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830
'''


import os
import logging
from pathlib import Path
import numpy as np
from ase.io import read
import sys

from functools import partial
from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core import pretrained_mlip

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sample', required=True, type=str, help='Sample file containing atomic configurations')
parser.add_argument('--savedir', required=True, type=str, help='Directory to save output')
parser.add_argument('--model-size', default="uma-m-1p1", type=str, help='UMA model')
parser.add_argument('--checkpoint', default=None, type=str, help='Path to checkpoint file')
parser.add_argument('--index', default=":", type=str, help='Configurations to load.')
parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"], help="Device to run inference on")
parser.add_argument('--batch-size', default=4, type=int, help='Batch size for inference')
parser.add_argument('--head', type=str, default='omat', choices=['oc20','omat','omol','odac','omc'], 
                    help='Model head to use: oc20 for catalysis, omat for inorganic materials, omol for molecules, odac for MOFs, omc for molecular crystals')
args = parser.parse_args()

os.makedirs(args.savedir, exist_ok=True)

sample = Path(args.sample).stem

if args.checkpoint is not None:
    args.model = args.checkpoint
else:
    args.model = args.model_size

def flatten(xss):
    # flatten list
    return [x for xs in xss for x in xs]

def collect(args: argparse.Namespace) -> None:
    sample = Path(args.sample).stem

    # Load model
    predictor = pretrained_mlip.get_predict_unit(args.model, device=args.device)
    predictor.model.eval()
    all_atom_refs = np.array(predictor.atom_refs[args.head])

    for param in predictor.model.parameters():
        param.requires_grad = False

    # Load data and prepare input
    atoms_list = read(args.sample, index=":")
    node_type = np.array(flatten([atoms.get_atomic_numbers() for atoms in atoms_list]))
    num_atoms = np.array([len(atoms) for atoms in atoms_list])
    atom_refs = all_atom_refs[node_type]

    # function to convert ase data
    a2g = partial(
                AtomicData.from_ase,
                task_name=args.head,
                r_edges=True,
                r_data_keys=["spin", "charge"],
                max_neigh=predictor.model.module.backbone.max_neighbors,
                radius=predictor.model.module.backbone.cutoff,
            )
    configs = [a2g(atoms) for atoms in atoms_list]

    # Collect data
    descriptors_list=[]
    node_energies_list=[]
    start=0
    for i in range(args.batch_size, len(configs)+args.batch_size, args.batch_size):
        batch = data_list_collater(configs[start:i], otf_graph=True)
        _=predictor.predict(batch)

        # L0 (invariant) node embeddings 
        node_features = predictor.model.module.backbone.forward(batch)['node_embedding']

        _input = node_features.narrow(1, 0, 1).squeeze(1)
        _output = predictor.model.module.output_heads.energyandforcehead.head.energy_block(_input)
        node_energy = _output.view(-1)

        descriptors_list.extend(node_features[:,0].detach().cpu())
        node_energies_list.append(node_energy.detach().cpu())

        start=i


    node_energies_list = np.concatenate(node_energies_list, axis=0)+atom_refs
    descriptors_list = np.vstack(descriptors_list)
    np.savez_compressed(os.path.join(args.savedir, f'embedding_info_{sample}.npz'), 
                        node_feats = descriptors_list, 
                        node_energies = node_energies_list, 
                        node_type = node_type, 
                        num_atoms = num_atoms)
    
    logging.info(f"UMA embeddings saved to {os.path.join(args.savedir, f'embedding_info_{sample}.npz')}.")
    
collect(args)

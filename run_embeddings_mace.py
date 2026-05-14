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

import numpy as np
from pathlib import Path
from ase.io import read
import os
import logging
from extract_mace_embeddings import InteractionHead

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sample', required=True, type=str, help='Sample file containing atomic configurations')
parser.add_argument('--savedir', required=True, type=str, help='Directory to save output')
parser.add_argument('--model-size', default="medium-0b", type=str, help='model size for MACE calculator')
parser.add_argument('--checkpoint', default=None, type=str, help='Optional path to finetuned model checkpoint')
parser.add_argument('--index', default=":", type=str, help='Configurations to load.')
args = parser.parse_args()

os.makedirs(args.savedir, exist_ok=True)

sample = Path(args.sample).stem

# Load structures using ASE
atoms_list = read(args.sample, index=args.index)  
logging.info(f'{len(atoms_list)} configurations loaded from {args.sample}.')

# Load MACE model 
head = InteractionHead(model=args.model_size, device="cuda", default_type="float32")

# Extract embeddings and node energies for each configuration
node_feats=[]
node_energies=[]
node_type = []
num_atoms = []
for atoms in atoms_list:
    out = head.forward(atoms)
    nf = out['node_feats'].cpu().detach().numpy()
    ne = out['node_energy'].cpu().detach().numpy()
    for i, elem in enumerate(atoms.get_atomic_numbers()):
        node_type.append(elem)
        node_energies.append(ne[i])
        node_feats.append(nf[i])
    num_atoms.append(len(atoms))
node_feats = np.vstack(node_feats)

# Save embeddings and energies to npz file
np.savez_compressed(os.path.join(args.savedir, f'embedding_info_{sample}.npz'), 
                    node_feats = node_feats, 
                    node_energies = node_energies, 
                    node_type = node_type, 
                    num_atoms=num_atoms)

logging.info(f"MACE embeddings saved to {os.path.join(args.savedir, f'embedding_info_{sample}.npz')}.")


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

from chgnet.graph import CrystalGraphConverter
from chgnet.model.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sample', required=True, type=str, help='Sample file containing atomic configurations')
parser.add_argument('--savedir', required=True, type=str, help='Directory to save output')
parser.add_argument('--model-size', default="0.3.0", type=str, help='CHGNet pretrained model name')
parser.add_argument('--checkpoint', default=None, type=str, help='Optional path to finetuned model checkpoint')
parser.add_argument('--index', default=":", type=str, help='Configurations to load.')
parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"], help="Device to run inference on")
parser.add_argument('--batch-size', default=16, type=int, help='Batch size for inference')
parser.add_argument('--vacuum', default=15.0, type=float, help='Vacuum padding (A) applied to non-periodic inputs')
parser.add_argument('--on-isolated-atoms', default='warn', choices=['ignore', 'warn', 'error'],
                    help='How CHGNet graph conversion handles atoms with no neighbor inside the cutoff')
args = parser.parse_args()

os.makedirs(args.savedir, exist_ok=True)

sample = Path(args.sample).stem

# Load structures using ASE
atoms_list = read(args.sample, index=args.index)
if not isinstance(atoms_list, list):
    atoms_list = [atoms_list]
logging.info(f'{len(atoms_list)} configurations loaded from {args.sample}.')

# Load CHGNet model
if args.checkpoint is not None:
    model = CHGNet.from_file(args.checkpoint).to(args.device)
else:
    model = CHGNet.load(model_name=args.model_size, use_device=args.device)
model.eval()
converter = CrystalGraphConverter(on_isolated_atoms=args.on_isolated_atoms)


def to_graph(atoms):
    # CHGNet is a crystal model and requires a periodic cell. Wrap non-periodic
    # inputs (e.g. isolated molecules) in a vacuum box so the structure can be
    # converted to a graph.
    if not bool(np.all(atoms.get_pbc())) or atoms.cell.rank < 3:
        atoms = atoms.copy()
        atoms.center(vacuum=args.vacuum)
        atoms.pbc = True
    return converter(AseAtomsAdaptor.get_structure(atoms))


graphs = [to_graph(atoms) for atoms in atoms_list]
predictions = model.predict_graph(
    graphs,
    task="e",
    return_site_energies=True,
    return_atom_feas=True,
    batch_size=args.batch_size,
)
if isinstance(predictions, dict):
    predictions = [predictions]

# Extract embeddings and node energies for each configuration
node_feats = []
node_energies = []
node_type = []
num_atoms = []
for atoms, prediction in zip(atoms_list, predictions):
    node_feats.append(np.asarray(prediction['atom_fea']))
    node_energies.append(np.asarray(prediction['site_energies']).reshape(-1))
    node_type.extend(atoms.get_atomic_numbers())
    num_atoms.append(len(atoms))
node_feats = np.vstack(node_feats)
node_energies = np.concatenate(node_energies)

# Save embeddings and energies to npz file
np.savez_compressed(os.path.join(args.savedir, f'embedding_info_{sample}.npz'),
                    node_feats = node_feats,
                    node_energies = node_energies,
                    node_type = node_type,
                    num_atoms = num_atoms)

logging.info(f"CHGNet embeddings saved to {os.path.join(args.savedir, f'embedding_info_{sample}.npz')}.")

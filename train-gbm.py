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
import logging
import os
from gbm import GBMRegressor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--embeddings', required=True, type=str, help='npz file containing pre-computed embeddings')
parser.add_argument('--savedir', required=True, type=str, help='Directory to save output')
parser.add_argument('--upper-alpha', default=0.95, type=float, help='Upper quantile for prediction intervals') 
parser.add_argument('--lower-alpha', default=0.05, type=float, help='Lower quantile for prediction intervals') 
parser.add_argument('--estimators', default=100, type=int, help='Number of estimators for the GBM') 
args = parser.parse_args()

os.makedirs(args.savedir, exist_ok=True)

# Load pre-computed node embeddings and node energies
data = np.load(args.embeddings)

node_feats = data['node_feats']
node_energies = data['node_energies']

logging.info(f"GBM training on {node_feats.shape[0]} nodes with {node_feats.shape[1]} features each.")

# Train GBM regressor on node-level data
gbm = GBMRegressor(savedir=args.savedir,
                   n_estimators=args.estimators, 
                   upper_alpha=args.upper_alpha, 
                   lower_alpha=args.lower_alpha)
gbm.update(node_feats, node_energies)

logging.info(f"Trained GBM saved to {args.savedir}.")


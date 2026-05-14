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
import pickle
import logging
import xgboost as xgb
import numpy as np

class GBMRegressor:
    """
    Union approach for Gradient Boosting Machine uncertainty estimation
    from https://link.springer.com/article/10.1186/s13321-023-00753-5 
    """
    def __init__(self, savedir='./', 
                 lower_alpha=0.1, 
                 upper_alpha=0.9, 
                 n_estimators=100,
                 learning_rate=0.04,
                 max_depth=5,
                 device='cpu',
                 ):
        """Initialize GBM regressors
        
        Args:
          savedir (str): Directory to save fit GBM regressors. 
                         (default: :obj:`./`)
          lower_alpha (float): The alpha-quantile of the quantile loss function.
                               Values must be in the range (0.0, 1.0). 
                               (default: :obj:`0.1`)
          upper_alpha (float): The alpha-quantile of the quantile loss function. 
                               Values must be in the range (0.0, 1.0). 
                               (default: :obj:`0.9`)
          n_estimators (int): The number of boosting stages to perform.
                              (default: :obj:`100`)
        """
        self.savedir = savedir
        self.alpha = np.array([lower_alpha,  upper_alpha])
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.hist = "gpu_hist" if device=='cuda' else "hist"
        
    @property
    def model_file(self):
        return f"GBMRegressor_{self.alpha[0]}-{self.alpha[-1]}.pkl"

    def update(self, embeddings, target):
        """Update GBM models after training epoch."""          
        Xy = xgb.QuantileDMatrix(embeddings, target)
        
        self.booster = xgb.train(
            {
                "objective": "reg:quantileerror",
                "tree_method": self.hist,
                "quantile_alpha": self.alpha,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "verbosity": 0,
                "disable_default_eval_metric": True,
            },
            Xy,
            num_boost_round=self.n_estimators,
            )

        # save updated model
        self._save()

    def forward(self, embeddings):
        """Predict quantiles for set of embeddings."""       
        scores = self.booster.inplace_predict(embeddings).T
        return scores

    def uncertainty(self, embeddings):
        """Return uncertainties for set of embeddings."""
        scores = self.forward(embeddings)
        return np.abs(scores[-1]-scores[0])/2

    def _save(self):
        """Save GBM regressor parameters to file."""
        with open(os.path.join(self.savedir, self.model_file), 'wb') as f:
            pickle.dump(self.booster, f)

        logging.info(f"Updated GBM regressor saved to {os.path.join(self.savedir, self.model_file)}")

    def _load(self):
        """Load trained GBM regressors from file."""
        if os.path.isfile(os.path.join(self.savedir, self.model_file)):
            with open(os.path.join(self.savedir, self.model_file), 'rb') as f:
                self.booster = pickle.load(f)
        else:
            logging.warning(f'No trained GBM regressor found in {self.savedir}. Call GBMRegressor.update to train a model.')
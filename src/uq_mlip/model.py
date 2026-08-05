"""Quantile GBM model used for per-atom UQ."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from uq_mlip.data import EmbeddingData, load_embeddings


class UQModel:
    """Train, load, and evaluate a quantile GBM UQ model."""

    def __init__(
        self,
        savedir: Union[str, Path],
        lower_alpha: float = 0.05,
        upper_alpha: float = 0.95,
        n_estimators: int = 100,
        learning_rate: float = 0.04,
        max_depth: int = 5,
        device: str = "cpu",
    ):
        self.savedir = Path(savedir)
        self.alpha = np.array([lower_alpha, upper_alpha])
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        # XGBoost >= 2.0 removed the "gpu_hist" tree method in favor of
        # tree_method="hist" with a separate device selector.
        self.tree_method = "hist"
        self.device = "cuda" if device == "cuda" else "cpu"
        self.booster = None

    @property
    def model_file(self) -> str:
        return f"GBMRegressor_{self.alpha[0]}-{self.alpha[-1]}.pkl"

    @property
    def model_path(self) -> Path:
        return self.savedir / self.model_file

    def fit(self, embeddings: EmbeddingData) -> "UQModel":
        """Fit the quantile model on per-atom embeddings and per-atom energies."""

        import xgboost as xgb

        embeddings.validate(require_energies=True)
        matrix = xgb.QuantileDMatrix(embeddings.node_feats, embeddings.node_energies)
        self.booster = xgb.train(
            {
                "objective": "reg:quantileerror",
                "tree_method": self.tree_method,
                "device": self.device,
                "quantile_alpha": self.alpha,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "verbosity": 0,
                "disable_default_eval_metric": True,
            },
            matrix,
            num_boost_round=self.n_estimators,
        )
        self.save()
        return self

    @classmethod
    def train_from_file(
        cls,
        embeddings_path: Union[str, Path],
        savedir: Union[str, Path],
        **kwargs,
    ) -> "UQModel":
        model = cls(savedir=savedir, **kwargs)
        return model.fit(load_embeddings(embeddings_path))

    def save(self) -> Path:
        if self.booster is None:
            raise RuntimeError("Cannot save UQModel before fitting or loading a booster.")
        self.savedir.mkdir(parents=True, exist_ok=True)
        with self.model_path.open("wb") as handle:
            pickle.dump(self.booster, handle)
        return self.model_path

    def load(self) -> "UQModel":
        if not self.model_path.is_file():
            raise FileNotFoundError(f"No trained UQ model found at {self.model_path}.")
        with self.model_path.open("rb") as handle:
            self.booster = pickle.load(handle)
        # Honor the requested device for prediction. A booster trained on GPU
        # otherwise stays pinned to "cuda" and warns when fed CPU-resident
        # arrays; defaulting to "cpu" matches the numpy inputs used here, while
        # device="cuda" opts back into GPU prediction.
        self.booster.set_param({"device": self.device})
        return self

    @classmethod
    def from_dir(cls, savedir: Union[str, Path], **kwargs) -> "UQModel":
        return cls(savedir=savedir, **kwargs).load()

    def predict_quantiles(self, node_feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.booster is None:
            raise RuntimeError("UQModel is not loaded or fitted.")
        scores = self.booster.inplace_predict(node_feats).T
        lower = scores[0]
        upper = scores[1]
        return lower, upper

    def uncertainty(self, node_feats: np.ndarray) -> np.ndarray:
        lower, upper = self.predict_quantiles(node_feats)
        return np.abs(upper - lower) / 2.0

    def predict_embeddings(self, embeddings: EmbeddingData) -> dict[str, np.ndarray]:
        embeddings.validate(require_energies=False)
        lower, upper = self.predict_quantiles(embeddings.node_feats)
        return {
            "lower": lower,
            "upper": upper,
            "uncertainty": np.abs(upper - lower) / 2.0,
        }

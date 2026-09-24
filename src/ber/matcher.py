"""Pair classifier with grouped cross-fitting.

Folds are drawn over Source 1 entities, so every candidate of an entity is
scored by a model that never saw that entity's labels. The resulting
out-of-fold probabilities are what the decision layer is tuned on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ber.config import ModelConfig

log = logging.getLogger(__name__)


def group_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold id per row; all rows of a group share a fold. Deterministic for a seed."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    fold_of = pd.Series(np.arange(len(shuffled)) % n_folds, index=shuffled)
    return fold_of.reindex(groups).to_numpy()


@dataclass
class Matcher:
    cfg: ModelConfig
    seed: int = 42
    feature_names: list[str] = field(default_factory=list)
    model: object | None = None
    best_iterations: list[int] = field(default_factory=list)
    fold_gain: list[np.ndarray] = field(default_factory=list)

    def _fit_one(
        self, x: pd.DataFrame, y: np.ndarray, x_val=None, y_val=None, rounds: int | None = None
    ):
        if self.cfg.kind == "lightgbm":
            import lightgbm as lgb

            params = {
                **self.cfg.params,
                "seed": self.seed,
                "deterministic": True,
                "force_row_wise": True,
            }
            train = lgb.Dataset(x, label=y, free_raw_data=False)
            callbacks = [lgb.log_evaluation(0)]
            valid_sets = []
            if x_val is not None and rounds is None:
                valid_sets = [lgb.Dataset(x_val, label=y_val, reference=train)]
                callbacks.append(lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False))
            booster = lgb.train(
                params,
                train,
                num_boost_round=rounds or self.cfg.num_boost_round,
                valid_sets=valid_sets or None,
                callbacks=callbacks,
            )
            best = booster.best_iteration or booster.current_iteration()
            return booster, int(best)
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            max_iter=rounds or self.cfg.num_boost_round,
            learning_rate=float(self.cfg.params.get("learning_rate", 0.05)),
            early_stopping=rounds is None,
            n_iter_no_change=self.cfg.early_stopping_rounds,
            random_state=self.seed,
        )
        clf.fit(x, y)
        return clf, int(clf.n_iter_)

    def _predict(self, model, x: pd.DataFrame) -> np.ndarray:
        if self.cfg.kind == "lightgbm":
            return model.predict(x, num_iteration=model.best_iteration or None)
        return model.predict_proba(x)[:, 1]

    def cross_fit(self, x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """Out-of-fold probabilities for every row."""
        self.feature_names = list(x.columns)
        folds = group_folds(groups, self.cfg.n_folds, self.seed)
        oof = np.zeros(len(x), dtype=np.float64)
        self.best_iterations = []
        self.fold_gain = []
        for k in range(self.cfg.n_folds):
            train_mask, valid_mask = folds != k, folds == k
            if not valid_mask.any():
                continue
            model, best = self._fit_one(x[train_mask], y[train_mask], x[valid_mask], y[valid_mask])
            oof[valid_mask] = self._predict(model, x[valid_mask])
            self.best_iterations.append(best)
            if self.cfg.kind == "lightgbm":
                self.fold_gain.append(model.feature_importance("gain"))
            log.info(
                "fold %d/%d: %d rows, best iteration %d",
                k + 1,
                self.cfg.n_folds,
                int(valid_mask.sum()),
                best,
            )
        return oof

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> Matcher:
        """Fit on everything with the round count the folds settled on."""
        self.feature_names = list(x.columns)
        rounds = None
        if self.best_iterations:
            rounds = max(1, int(np.mean(self.best_iterations) * self.cfg.final_rounds_multiplier))
        self.model, _ = self._fit_one(x, y, rounds=rounds or self.cfg.num_boost_round)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("matcher is not fitted")
        missing = [c for c in self.feature_names if c not in x.columns]
        if missing:
            raise ValueError(f"feature frame lacks {missing}")
        return self._predict(self.model, x[self.feature_names])

    def feature_importance(self) -> pd.DataFrame:
        """Gain importance of the final model, or the fold average before the final fit."""
        if self.cfg.kind != "lightgbm":
            return pd.DataFrame({"feature": self.feature_names})
        if self.model is not None:
            gain = self.model.feature_importance("gain")
        elif self.fold_gain:
            gain = np.mean(self.fold_gain, axis=0)
        else:
            return pd.DataFrame({"feature": self.feature_names})
        frame = pd.DataFrame({"feature": self.feature_names, "gain": gain})
        frame["gain_share"] = frame["gain"] / max(frame["gain"].sum(), 1e-12)
        return frame.sort_values("gain", ascending=False).reset_index(drop=True)

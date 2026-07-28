"""
Model training: Random Forest and XGBoost over stylometric features, plus an
optional fine-tuned transformer baseline.

This module REQUIRES scikit-learn (and optionally xgboost / transformers), which
you install from requirements.txt in your run environment (laptop or Colab).
Feature extraction (src/features.py) is dependency-light on purpose; the heavy
ML dependencies live here.

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class TrainedModel:
    name: str
    model: Any
    scaler: Optional[Any]
    best_params: Dict[str, Any]


def _standardise(scaler, X):
    return X if scaler is None else scaler.transform(X)


def train_random_forest(X_train, y_train, seed: int = 42) -> TrainedModel:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "class_weight": ["balanced"],
    }
    base = RandomForestClassifier(random_state=seed, n_jobs=-1)
    search = GridSearchCV(base, grid, scoring="f1", cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    return TrainedModel("random_forest", search.best_estimator_, None, search.best_params_)


def train_xgboost(X_train, y_train, seed: int = 42) -> TrainedModel:
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV

    pos = float(np.sum(y_train == 0)) / max(1.0, float(np.sum(y_train == 1)))
    grid = {
        "n_estimators": [300, 600],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    base = XGBClassifier(
        random_state=seed, n_jobs=-1, eval_metric="logloss",
        scale_pos_weight=pos, tree_method="hist",
    )
    search = GridSearchCV(base, grid, scoring="f1", cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    return TrainedModel("xgboost", search.best_estimator_, None, search.best_params_)


def train_logreg_baseline(X_train, y_train, seed: int = 42) -> TrainedModel:
    """A standardised logistic-regression baseline - cheap, interpretable, and a
    sanity floor for the tree models."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(scaler.transform(X_train), y_train)
    return TrainedModel("logreg", clf, scaler, {})


def predict(tm: TrainedModel, X) -> np.ndarray:
    return tm.model.predict(_standardise(tm.scaler, X))


def predict_proba(tm: TrainedModel, X) -> np.ndarray:
    Xs = _standardise(tm.scaler, X)
    if hasattr(tm.model, "predict_proba"):
        return tm.model.predict_proba(Xs)[:, 1]
    return tm.model.predict(Xs).astype(float)


# --- Optional transformer baseline (Should-have; first item cut under compute
#     pressure per the proposal's contingency plan). Kept minimal on purpose. ---
def train_transformer_baseline(train_texts, y_train, val_texts, y_val,
                               model_name: str = "distilbert-base-uncased",
                               epochs: int = 3, seed: int = 42):
    """Fine-tune a small transformer as a comparison baseline. Requires the
    'transformers' and 'torch' extras. Returns an object exposing .predict(texts).
    Implementation intentionally compact; see report Appendix for full code."""
    raise NotImplementedError(
        "Transformer baseline is an optional extra. Enable by installing the "
        "'transformers' and 'torch' packages and completing this function in your "
        "run environment. The stylometric models above already answer the RQ."
    )

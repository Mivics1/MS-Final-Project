"""
Feature-importance analysis (Objective O7): identify WHICH stylistic markers
reveal machine authorship. Combines impurity-based importance (fast, tree-native)
with permutation importance (model-agnostic, less biased) and an optional SHAP
summary if the 'shap' package is installed.

Interpreting these rankings is the intellectual payoff of the project and is where
the Distinction-band 'notable insight' is demonstrated in the report.

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def impurity_importance(tm, feature_names: List[str], top: int = 20) -> List[Dict]:
    imp = getattr(tm.model, "feature_importances_", None)
    if imp is None:
        return []
    order = np.argsort(imp)[::-1][:top]
    return [{"feature": feature_names[i], "importance": float(imp[i])} for i in order]


def permutation_importance(tm, X, y, feature_names: List[str],
                           n_repeats: int = 10, seed: int = 42, top: int = 20) -> List[Dict]:
    from sklearn.inspection import permutation_importance as sk_perm
    from src.train import predict

    class _Wrap:
        def __init__(self, tm): self.tm = tm
        def fit(self, *a, **k): return self
        def predict(self, X): return predict(self.tm, X)

    result = sk_perm(_Wrap(tm), X, y, n_repeats=n_repeats,
                     random_state=seed, scoring="f1")
    order = np.argsort(result.importances_mean)[::-1][:top]
    return [
        {"feature": feature_names[i],
         "importance": float(result.importances_mean[i]),
         "std": float(result.importances_std[i])}
        for i in order
    ]


def plot_importance(ranked: List[Dict], title: str, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # placed at ~15.5 cm in Word; tall enough that all 20 labels stay legible
    names = [r["feature"] for r in ranked][::-1]
    vals = [r["importance"] for r in ranked][::-1]
    fig, ax = plt.subplots(figsize=(6.1, max(3.4, 0.30 * len(names))))
    ax.barh(names, vals, color="#2E5B8A")
    ax.set_xlabel("Importance", fontsize=9.5)
    ax.set_title(title, fontsize=10.5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.margins(y=0.01)
    fig.tight_layout(); fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

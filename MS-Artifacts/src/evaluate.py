"""
Evaluation and reporting: metrics, confusion matrices, ROC/PR curves, and a
results table ready to paste into the dissertation's Evaluation chapter.

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np


def evaluate_model(tm, X_test, y_test) -> Dict:
    """Compute the full metric suite for one trained model on the test set."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score,
        confusion_matrix, roc_auc_score,
    )
    from src.train import predict, predict_proba
    from src.metrics import bootstrap_f1_ci

    y_pred = predict(tm, X_test)
    try:
        y_prob = predict_proba(tm, X_test)
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = float("nan")

    ci = bootstrap_f1_ci(y_test, y_pred)
    return {
        "model": tm.name,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": auc,
        "f1_ci_low": ci["ci_low"],
        "f1_ci_high": ci["ci_high"],
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "best_params": tm.best_params,
    }


def results_table(results: List[Dict]) -> str:
    """Render a markdown table of model results (for the report / appendix)."""
    header = "| Model | Precision | Recall | F1 | F1 95% CI | ROC-AUC | Accuracy |\n"
    header += "|---|---|---|---|---|---|---|\n"
    rows = ""
    for r in results:
        rows += (
            f"| {r['model']} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1']:.3f} | [{r['f1_ci_low']:.3f}, {r['f1_ci_high']:.3f}] | "
            f"{r['roc_auc']:.3f} | {r['accuracy']:.3f} |\n"
        )
    return header + rows


def save_results(results: List[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    with open(os.path.join(out_dir, "results_table.md"), "w") as fh:
        fh.write(results_table(results))
    print(f"[evaluate] wrote results to {out_dir}")


def plot_confusion(cm, model_name: str, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # placed at ~11.5 cm in Word; 4.5 in at 400 dpi = 1,800 px
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.imshow(cm, cmap="Blues")
    labels = ["Real-world", "AI-generated"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=9, rotation=90, va="center")
    ax.set_xlabel("Predicted class", fontsize=9.5)
    ax.set_ylabel("Actual class", fontsize=9.5)
    ax.set_title(f"Confusion matrix — {model_name}", fontsize=10.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=15,
                    weight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "#1F2A37")
    fig.tight_layout(); fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

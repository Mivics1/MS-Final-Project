"""
Smoke test: verify feature extraction + a numpy-only classifier + metrics run
end-to-end WITHOUT scikit-learn. Uses the neutral sample data.

This proves the plumbing works in a minimal environment. The real pipeline
(src/pipeline.py) uses scikit-learn / xgboost for the reported results.

Run:  python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import features               # noqa: E402
from src.metrics import binary_metrics, bootstrap_f1_ci, confusion_matrix  # noqa: E402
from scripts.make_sample_data import build  # noqa: E402


def _standardise(X, mean, std):
    return (X - mean) / np.where(std == 0, 1.0, std)


def train_logreg_numpy(X, y, lr=0.1, epochs=400, seed=42):
    """Minimal L2 logistic regression in numpy - only to exercise the flow."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0, 0.01, d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        gw = X.T @ (p - y) / n + 1e-3 * w
        gb = float(np.mean(p - y))
        w -= lr * gw
        b -= lr * gb
    return w, b


def main():
    df = build(n_per_class=150)
    texts = df["text"].tolist()
    y = df["label"].to_numpy()

    # same extraction code as production pipeline
    X = features.extract_matrix(texts)
    assert X.shape[0] == len(texts)
    assert X.shape[1] == len(features.FEATURE_NAMES)
    print(f"[smoke] extracted feature matrix: {X.shape}")

    # deterministic split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]

    mean, std = X[tr].mean(0), X[tr].std(0)
    Xtr, Xte = _standardise(X[tr], mean, std), _standardise(X[te], mean, std)

    w, b = train_logreg_numpy(Xtr, y[tr])
    prob = 1 / (1 + np.exp(-(Xte @ w + b)))
    pred = (prob >= 0.5).astype(int)

    m = binary_metrics(y[te], pred)
    ci = bootstrap_f1_ci(y[te], pred, n_boot=500)
    cm = confusion_matrix(y[te], pred)

    print(f"[smoke] confusion matrix [[TN,FP],[FN,TP]] = {cm.tolist()}")
    print(f"[smoke] precision={m['precision']:.3f} recall={m['recall']:.3f} "
          f"f1={m['f1']:.3f} acc={m['accuracy']:.3f}")
    print(f"[smoke] bootstrap F1 95% CI = [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")

    assert m["f1"] > 0.8, "machinery check: neutral classes should separate easily"
    print("\n[smoke] PASS - feature extraction, training and metrics all execute. "
          "(Neutral data; NOT dissertation results.)")


if __name__ == "__main__":
    main()

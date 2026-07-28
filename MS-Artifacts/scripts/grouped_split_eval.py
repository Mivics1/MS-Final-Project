"""
Campaign-level leakage control and a 2022-only sensitivity analysis
(addresses reviewer points 3, 5 and 6).

1. GROUPED SPLIT. Phishing campaigns repeat a template with small edits, so a
   random split can place near-identical variants in both train and test and
   inflate the metrics. Here every email is clustered by textual similarity
   (character 5-gram shingles, Jaccard >= 0.5, single-linkage via union-find),
   and the train/test split is made over WHOLE CLUSTERS so no cluster spans the
   partitions. Models are retrained with the tuned params from the main run and
   re-evaluated. If performance holds up under this stricter split, the result
   is not an artefact of template leakage.

2. 2022-ONLY SENSITIVITY. The real-world (Nazario) class cannot be verified as
   human-authored, and 2023-24 overlaps heavy public LLM use. Restricting that
   class to the 2022 archive (the least-contaminated year available) and
   rebalancing tests whether the signal survives on the most defensible subset.

Run:  PYTHONPATH=. .venv/bin/python scripts/grouped_split_eval.py
Outputs: results/grouped_split.json

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src import data_loader, features, train
from src.metrics import bootstrap_f1_ci

SEED = 42
TEST_SIZE = 0.15
JACCARD = 0.5


def shingles(text: str, k: int = 5) -> set:
    s = re.sub(r"[^a-z0-9]", "", text.lower())
    if len(s) < k:
        return {s} if s else set()
    return {s[i:i + k] for i in range(len(s) - k + 1)}


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def cluster(texts: list[str], thr: float = JACCARD) -> np.ndarray:
    """Single-linkage clustering by Jaccard over char-5-gram shingles.
    Bucketed by shingle so we only compare emails that share a shingle."""
    shs = [shingles(t) for t in texts]
    uf = UnionFind(len(texts))
    buckets: dict = {}
    for i, sh in enumerate(shs):
        for g in sh:
            buckets.setdefault(g, []).append(i)
    checked = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a_idx in range(len(members)):
            i = members[a_idx]
            for b_idx in range(a_idx + 1, len(members)):
                j = members[b_idx]
                if (i, j) in checked:
                    continue
                checked.add((i, j))
                inter = len(shs[i] & shs[j])
                if inter == 0:
                    continue
                union = len(shs[i] | shs[j])
                if union and inter / union >= thr:
                    uf.union(i, j)
    return np.array([uf.find(i) for i in range(len(texts))])


def grouped_split(df: pd.DataFrame, clusters: np.ndarray, seed=SEED):
    """Assign whole clusters to train/test, keeping class balance approximately.
    Clusters are class-pure in practice (a template is one class)."""
    rng = np.random.default_rng(seed)
    by_cluster = {}
    for idx, c in enumerate(clusters):
        by_cluster.setdefault(c, []).append(idx)
    cids = list(by_cluster)
    # cluster label = majority class of its members
    clab = {c: int(round(np.mean(df["label"].to_numpy()[by_cluster[c]]))) for c in cids}
    test_idx, train_idx = [], []
    for cls in (0, 1):
        cls_clusters = [c for c in cids if clab[c] == cls]
        rng.shuffle(cls_clusters)
        n_test = int(round(sum(len(by_cluster[c]) for c in cls_clusters) * TEST_SIZE))
        taken = 0
        for c in cls_clusters:
            if taken < n_test:
                test_idx += by_cluster[c]
                taken += len(by_cluster[c])
            else:
                train_idx += by_cluster[c]
    return np.array(train_idx), np.array(test_idx)


def fit_models(Xtr, ytr, rf_params, xgb_params):
    """Fit all three classifiers with the hyperparameters selected in the main
    run, so every robustness condition covers the same models as Table 6.1."""
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    models = [("logreg", train.train_logreg_baseline(Xtr, ytr, SEED))]
    models.append(("random_forest", train.TrainedModel(
        "random_forest",
        RandomForestClassifier(random_state=SEED, n_jobs=2, **rf_params).fit(Xtr, ytr),
        None, rf_params)))
    pos = float(np.sum(ytr == 0)) / max(1.0, float(np.sum(ytr == 1)))
    models.append(("xgboost", train.TrainedModel(
        "xgboost",
        XGBClassifier(random_state=SEED, n_jobs=2, eval_metric="logloss",
                      scale_pos_weight=pos, tree_method="hist",
                      **xgb_params).fit(Xtr, ytr),
        None, xgb_params)))
    return models


def evaluate_on(df, train_idx, test_idx, rf_params, xgb_params):
    from sklearn.metrics import f1_score, roc_auc_score
    texts = df["text"].tolist()
    y = df["label"].to_numpy()
    Xtr = features.extract_matrix([texts[i] for i in train_idx])
    Xte = features.extract_matrix([texts[i] for i in test_idx])
    ytr, yte = y[train_idx], y[test_idx]
    out = {"n_train": int(len(ytr)), "n_test": int(len(yte)),
           "test_positive": int(np.sum(yte == 1)),
           "test_negative": int(np.sum(yte == 0)),
           "seed": SEED}
    for name, m in fit_models(Xtr, ytr, rf_params, xgb_params):
        yp = train.predict(m, Xte)
        ci = bootstrap_f1_ci(yte, yp)
        out[name] = {
            "f1": float(f1_score(yte, yp)),
            "f1_ci": [ci["ci_low"], ci["ci_high"]],
            "roc_auc": float(roc_auc_score(yte, train.predict_proba(m, Xte))),
        }
    return out


def main():
    df = data_loader.load_csv("data/emails.csv")
    _res = json.load(open("results/results.json"))
    rf_params = next(r["best_params"] for r in _res if r["model"] == "random_forest")
    xgb_params = next(r["best_params"] for r in _res if r["model"] == "xgboost")

    texts = df["text"].tolist()
    clusters = cluster(texts)
    _, counts = np.unique(clusters, return_counts=True)
    cluster_stats = {
        "n_emails": int(len(df)),
        "n_clusters": int(len(counts)),
        "singletons": int(np.sum(counts == 1)),
        "largest_cluster": int(counts.max()),
        "emails_in_multi_clusters": int(np.sum(counts[counts > 1])),
        "jaccard_threshold": JACCARD,
    }

    tr, te = grouped_split(df, clusters)
    grouped = evaluate_on(df, tr, te, rf_params, xgb_params)

    # 2022-only sensitivity: rebuild a balanced set using only 2022 real-world
    src = pd.read_csv("data/emails.csv")
    naz22 = src[src["source"] == "phishing-2022"]
    ai = src[src["label"] == 1]
    m = min(len(naz22), len(ai))
    rng = np.random.default_rng(SEED)
    sub = pd.concat([
        naz22.sample(m, random_state=SEED),
        ai.sample(m, random_state=SEED),
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    split = data_loader.stratified_split(sub, seed=SEED)
    Xtr = features.extract_matrix(split.X_train_text)
    Xte = features.extract_matrix(split.X_test_text)
    from sklearn.metrics import f1_score, roc_auc_score
    s2022 = {"n_per_class": int(m),
             "n_train": int(len(split.y_train)),
             "n_test": int(len(split.y_test)),
             "test_positive": int(np.sum(split.y_test == 1)),
             "test_negative": int(np.sum(split.y_test == 0)),
             "seed": SEED}
    for name, mdl in fit_models(Xtr, split.y_train, rf_params, xgb_params):
        yp = train.predict(mdl, Xte)
        ci = bootstrap_f1_ci(split.y_test, yp)
        s2022[name] = {"f1": float(f1_score(split.y_test, yp)),
                       "f1_ci": [ci["ci_low"], ci["ci_high"]],
                       "roc_auc": float(roc_auc_score(split.y_test,
                                                      train.predict_proba(mdl, Xte)))}

    result = {"cluster_stats": cluster_stats,
              "grouped_split": grouped,
              "sensitivity_2022_only": s2022}
    with open("results/grouped_split.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

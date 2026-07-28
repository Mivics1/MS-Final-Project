"""
Supplementary analysis for the Evaluation chapter.

1. Per-class means for the most important features (direction of the signal).
2. Random-Forest impurity + permutation importance (the pipeline interprets the
   best-F1 model, which turned out to be logreg; RF gives the tree-native view)
   and the feature_importance.png figure.
3. Artefact ablation: retrain after dropping email_addr_count and url_count,
   the two features most exposed to a corpus artefact (real recipient
   addresses in the Nazario mboxes vs fabricated URLs in the AI corpus).

Run:  .venv/bin/python scripts/supplementary_analysis.py
Outputs: results/supplementary.json, results/feature_importance.png,
         results/class_means.md

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

import json

import numpy as np

from src import data_loader, features, interpret, train
from src.metrics import bootstrap_f1_ci

SEED = 42
DROP = ["email_addr_count", "url_count"]


def main() -> None:
    df = data_loader.load_csv("data/emails.csv")
    split = data_loader.stratified_split(df, seed=SEED)
    names = features.FEATURE_NAMES
    X_train = features.extract_matrix(split.X_train_text)
    X_test = features.extract_matrix(split.X_test_text)

    out = {}

    # 1. per-class feature means on the full corpus
    X_all = features.extract_matrix(df["text"].tolist())
    y_all = df["label"].to_numpy()
    focus = ["exclamation_ratio", "email_addr_count", "uppercase_ratio",
             "avg_word_length", "period_ratio", "digit_ratio", "url_count",
             "urgency_word_ratio", "word_count", "std_sentence_length",
             "type_token_ratio", "flesch_reading_ease", "flesch_kincaid_grade",
             "fw_the", "fw_do", "comma_ratio"]
    means = {}
    for f in focus:
        i = names.index(f)
        means[f] = {
            "human_mean": float(np.mean(X_all[y_all == 0, i])),
            "llm_mean": float(np.mean(X_all[y_all == 1, i])),
        }
    out["class_means"] = means
    with open("results/class_means.md", "w") as fh:
        fh.write("| Feature | Human mean | LLM mean |\n|---|---|---|\n")
        for f, v in means.items():
            fh.write(f"| {f} | {v['human_mean']:.4f} | {v['llm_mean']:.4f} |\n")

    # 2. RF importance + figure (single fit with the tuned params from the main
    #    run, rather than a second full grid search, to stay within memory)
    from sklearn.ensemble import RandomForestClassifier
    best = json.load(open("results/results.json"))
    rf_params = next(r["best_params"] for r in best if r["model"] == "random_forest")
    rf_clf = RandomForestClassifier(random_state=SEED, n_jobs=2, **rf_params)
    rf_clf.fit(X_train, split.y_train)
    rf = train.TrainedModel("random_forest", rf_clf, None, rf_params)
    imp = interpret.impurity_importance(rf, names, top=20)
    interpret.plot_importance(imp, "Top features - random_forest (impurity)",
                              "results/feature_importance.png")
    perm = interpret.permutation_importance(rf, X_test, split.y_test, names,
                                            n_repeats=10, top=20)
    out["rf_impurity_top20"] = imp
    out["rf_permutation_top20"] = perm

    # rank-agreement between the two importance methods (reviewer point 10):
    # Spearman correlation over the union of each method's top-20 features
    from scipy.stats import spearmanr
    imp_rank = {r["feature"]: i for i, r in enumerate(imp)}
    perm_rank = {r["feature"]: i for i, r in enumerate(perm)}
    common = [fn for fn in imp_rank if fn in perm_rank]
    rho, pval = spearmanr([imp_rank[fn] for fn in common],
                          [perm_rank[fn] for fn in common])
    out["importance_agreement"] = {
        "features_in_both_top20": len(common),
        "spearman_rho": float(rho),
        "p_value": float(pval),
    }

    # 3. ablation without the artefact-exposed structural features
    keep = [i for i, n in enumerate(names) if n not in DROP]
    Xtr_a, Xte_a = X_train[:, keep], X_test[:, keep]
    abl = {"n_train": int(len(split.y_train)), "n_test": int(len(split.y_test)),
           "seed": SEED}
    xgb_params = next(r["best_params"] for r in best if r["model"] == "xgboost")

    def rf_fixed(X, y, seed):
        clf = RandomForestClassifier(random_state=seed, n_jobs=2, **rf_params)
        clf.fit(X, y)
        return train.TrainedModel("random_forest", clf, None, rf_params)

    def xgb_fixed(X, y, seed):
        from xgboost import XGBClassifier
        pos = float(np.sum(y == 0)) / max(1.0, float(np.sum(y == 1)))
        clf = XGBClassifier(random_state=seed, n_jobs=2, eval_metric="logloss",
                            scale_pos_weight=pos, tree_method="hist", **xgb_params)
        clf.fit(X, y)
        return train.TrainedModel("xgboost", clf, None, xgb_params)

    for name, fn in [("logreg", train.train_logreg_baseline),
                     ("random_forest", rf_fixed),
                     ("xgboost", xgb_fixed)]:
        m = fn(Xtr_a, split.y_train, SEED)
        y_pred = train.predict(m, Xte_a)
        from sklearn.metrics import f1_score, roc_auc_score
        ci = bootstrap_f1_ci(split.y_test, y_pred)
        abl[name] = {
            "f1": float(f1_score(split.y_test, y_pred)),
            "f1_ci": [ci["ci_low"], ci["ci_high"]],
            "roc_auc": float(roc_auc_score(split.y_test,
                                           train.predict_proba(m, Xte_a))),
        }
    out["ablation_dropped"] = DROP
    out["ablation"] = abl

    with open("results/supplementary.json", "w") as fh:
        json.dump(out, fh, indent=2)

    # structured exports so the document builder and the validator read tables
    # from data rather than from hand-typed values
    import csv as _csv
    with open("results/feature_means.csv", "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["feature", "real_world_mean", "ai_mean"])
        for feat, v in means.items():
            w.writerow([feat, f"{v['human_mean']:.6f}", f"{v['llm_mean']:.6f}"])
    with open("results/impurity_importance.csv", "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["rank", "feature", "importance"])
        for i, r in enumerate(imp, 1):
            w.writerow([i, r["feature"], f"{r['importance']:.6f}"])
    with open("results/permutation_importance.csv", "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["rank", "feature", "importance", "std"])
        for i, r in enumerate(perm, 1):
            w.writerow([i, r["feature"], f"{r['importance']:.6f}", f"{r['std']:.6f}"])
    print(json.dumps({"ablation": abl}, indent=2))
    print("top RF impurity:", [r["feature"] for r in imp[:10]])
    print("class means written to results/class_means.md")


if __name__ == "__main__":
    main()

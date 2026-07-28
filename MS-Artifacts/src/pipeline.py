"""
End-to-end orchestration: data -> features -> train -> evaluate -> interpret.

Run:
    python -m src.pipeline --data data/emails.csv --out results/

This is the single entry point that reproduces every number in the dissertation's
Evaluation chapter from the raw CSV. Reproducibility is guaranteed by the fixed
--seed (default 42) threaded through splitting and every model.

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

import argparse
import os

from src import data_loader, evaluate, features, interpret, train


def run(data_path: str, out_dir: str, seed: int = 42, with_xgb: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load, clean, split
    df = data_loader.load_csv(data_path)
    print("[pipeline] class balance:", data_loader.class_balance(df["label"].to_numpy()))
    split = data_loader.stratified_split(df, seed=seed)

    # 2. Feature extraction (same code path as the smoke test)
    X_train = features.extract_matrix(split.X_train_text)
    X_val = features.extract_matrix(split.X_val_text)
    X_test = features.extract_matrix(split.X_test_text)
    names = features.FEATURE_NAMES
    print(f"[pipeline] extracted {X_train.shape[1]} features")

    # 3. Train models
    models = [
        train.train_logreg_baseline(X_train, split.y_train, seed),
        train.train_random_forest(X_train, split.y_train, seed),
    ]
    if with_xgb:
        try:
            models.append(train.train_xgboost(X_train, split.y_train, seed))
        except Exception as e:  # xgboost optional
            print("[pipeline] skipping xgboost:", e)

    # 4. Evaluate
    results = [evaluate.evaluate_model(m, X_test, split.y_test) for m in models]
    evaluate.save_results(results, out_dir)
    print("\n" + evaluate.results_table(results))
    for r in results:
        evaluate.plot_confusion(
            r["confusion_matrix"], r["model"],
            os.path.join(out_dir, f"confusion_{r['model']}.png"),
        )

    # 5. Interpret the best model (highest F1)
    best = max(zip(results, models), key=lambda z: z[0]["f1"])[1]
    imp = interpret.impurity_importance(best, names, top=20)
    if imp:
        interpret.plot_importance(imp, f"Top features - {best.name}",
                                  os.path.join(out_dir, "feature_importance.png"))
    perm = interpret.permutation_importance(best, X_test, split.y_test, names, top=20)
    with open(os.path.join(out_dir, "feature_importance.md"), "w") as fh:
        fh.write(f"# Feature importance - {best.name}\n\n## Impurity-based\n\n")
        for r in imp:
            fh.write(f"- {r['feature']}: {r['importance']:.4f}\n")
        fh.write("\n## Permutation-based\n\n")
        for r in perm:
            fh.write(f"- {r['feature']}: {r['importance']:.4f} (±{r['std']:.4f})\n")
    print("[pipeline] done.")


def main():
    ap = argparse.ArgumentParser(description="Stylometric LLM-phishing detection pipeline")
    ap.add_argument("--data", required=True, help="CSV with columns text,label")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-xgb", action="store_true", help="skip xgboost")
    args = ap.parse_args()
    run(args.data, args.out, seed=args.seed, with_xgb=not args.no_xgb)


if __name__ == "__main__":
    main()

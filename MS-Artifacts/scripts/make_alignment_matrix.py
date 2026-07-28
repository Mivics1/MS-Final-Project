"""
Generate results/proposal_alignment_matrix.csv.

Maps each commitment in MS_Project_Proposal.docx to what the finalised study
actually implemented, with the artefact evidence for the claim. Status values
are restricted to the four permitted terms.

Run:  PYTHONPATH=. .venv/bin/python scripts/make_alignment_matrix.py

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

import csv
import json
import os

RESULTS = "results"

CONCLUSION = (
    "The dissertation remains aligned with the proposal's central topic, attack "
    "scope, stylometric methodology, primary classifiers, evaluation target and "
    "interpretability deliverable. The use of a published AI-generated dataset, "
    "relabelling of the Nazario class and omission of the transformer are "
    "disclosed methodological refinements rather than a change of topic."
)


def main() -> None:
    summary = json.load(open(os.path.join("data", "corpus_summary.json")))
    results = {r["model"]: r for r in json.load(
        open(os.path.join(RESULTS, "results.json")))}
    grouped = json.load(open(os.path.join(RESULTS, "grouped_split.json")))
    supp = json.load(open(os.path.join(RESULTS, "supplementary.json")))

    n = summary["per_class_after_balancing"]
    lr, rf, xgb = results["logreg"], results["random_forest"], results["xgboost"]

    rows = [
        ("Central research question: can stylometric and lexical features "
         "distinguish AI-generated from other credential-harvesting phishing",
         "Retained. The final study measures the same question against real-world "
         "Nazario phishing rather than verified human-written phishing",
         "Fulfilled with justified refinement",
         "Dissertation title, aim and Chapter 1; Chapters 6-7",
         "The Nazario corpus is hand-classified as phishing but not verified for "
         "authorship, so the class is labelled 'real-world' to match the evidence"),

        ("Single attack type: credential-harvesting emails",
         "Enforced by a documented, reproducible two-condition screen applied "
         "identically to both sources",
         "Fulfilled with justified refinement",
         f"scripts/build_corpus.py; Table B.1; "
         f"{summary['funnel']['nazario']['excluded_not_credential_harvesting']} "
         f"real-world and "
         f"{summary['funnel']['eze_shamir']['excluded_not_credential_harvesting']} "
         "AI messages excluded by the screen",
         "The proposal asserted the scope; the final study evidences it and audits "
         "the screen, reporting exploratory precision and recall"),

        ("Balanced binary corpus",
         f"{summary['total']} messages, {n} per class, exact duplicates removed "
         "before splitting",
         "Fulfilled",
         "data/corpus_summary.json; Table B.1; validation checks 1-2",
         "Balanced by seeded down-sampling of the larger class"),

        ("LLM phishing generated locally from a single model family under "
         "controlled prompts",
         "Replaced by the published Eze and Shamir (2024) dataset produced via the "
         "DeepAI text-generation service",
         "Fulfilled with justified refinement",
         "Sections 3.2, 3.7 and 7.3; Table B.2 provenance and checksums",
         "Removes the dual-use hazard of generating new attack content and makes "
         "the corpus citable and reproducible; the exact underlying model is not "
         "established by the source, which is disclosed as a limitation"),

        ("Human-written phishing drawn from established public corpora",
         "Nazario phishing corpus 2022-2024 under CC BY 4.0, relabelled "
         "'real-world phishing'",
         "Fulfilled with justified refinement",
         "Section 3.2; Table B.2; Nazario reference entry",
         "Authorship is not independently verified and the archives overlap public "
         "LLM availability, so no human-authorship claim is made"),

        ("Stylometric and lexical feature extraction",
         "95 interpretable features in six groups, standard library and numpy only",
         "Fulfilled",
         "src/features.py; Table 4.1",
         "Perplexity was not used because it requires access to a generating model, "
         "which the defender scenario excludes"),

        ("Random Forest classifier",
         f"Trained with five-fold cross-validated grid search; test F1 {rf['f1']:.3f}",
         "Fulfilled",
         "results/results.json; Table 6.1",
         ""),

        ("XGBoost classifier",
         f"Trained with five-fold cross-validated grid search; test F1 {xgb['f1']:.3f}",
         "Fulfilled",
         "results/results.json; Table 6.1",
         ""),

        ("Fine-tuned transformer comparison baseline",
         "Not implemented",
         "Removed under proposal contingency",
         "Section 5.6; src/train.py documents the unused entry point",
         "The proposal named the transformer as the first item to cut if compute ran "
         "short; effort moved to corpus validity and robustness testing"),

        ("Evaluate precision, recall and F1 on a held-out test set",
         "Reported for all three classifiers, with accuracy, ROC-AUC and percentile "
         "bootstrap 95% confidence intervals",
         "Expanded beyond proposal",
         "Table 6.1; validation checks recompute every metric from the confusion "
         "matrices",
         "Confidence intervals and a logistic-regression floor were added beyond the "
         "proposal's metric list"),

        ("Held-out F1 of at least 0.90",
         f"Met by every classifier: logistic {lr['f1']:.3f}, XGBoost {xgb['f1']:.3f}, "
         f"Random Forest {rf['f1']:.3f}",
         "Fulfilled",
         "Table 6.1",
         "Target was pre-specified in the approved proposal, not formally "
         "preregistered, and is described as such"),

        ("Rank feature importance to identify discriminative stylistic markers",
         "Impurity and permutation rankings both reported, with their rank agreement "
         f"({supp['importance_agreement']['features_in_both_top20']} shared features, "
         f"Spearman rho = {supp['importance_agreement']['spearman_rho']:.2f})",
         "Expanded beyond proposal",
         "Tables 6.2-6.3; results/impurity_importance.csv; "
         "results/permutation_importance.csv",
         "Two methods are reported rather than one because they disagree on ordering"),

        ("Reproducible pipeline",
         "Seed-fixed end-to-end scripts, source checksums, per-stage corpus funnel "
         "and an automated alignment validator",
         "Expanded beyond proposal",
         "Appendix A; results/final_alignment_validation.json",
         "Robustness analyses added beyond the proposal: structural-cue ablation, "
         "similarity-cluster grouped partition and a 2022-only sensitivity check "
         f"(grouped test n = {grouped['grouped_split']['n_test']}, "
         f"2022 test n = {grouped['sensitivity_2022_only']['n_test']})"),

        ("Dual-use safeguards and ethical handling",
         "No message sent, no new phishing generated, no raw phishing text published, "
         "encrypted storage, deletion at project end",
         "Fulfilled",
         "Section 3.6; repository .gitignore excludes data/raw and the audit sample",
         "Switching to a published corpus reduced the dual-use footprint relative to "
         "the proposal"),

        ("Two-week project timeline with supervisor checkpoints",
         "Delivered; effort shifted from the transformer baseline to corpus screening, "
         "auditing and robustness testing",
         "Fulfilled with justified refinement",
         "Section 7.4",
         "Corpus construction became the critical path once the screen defect was "
         "found and corrected"),
    ]

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "proposal_alignment_matrix.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Proposal commitment", "Final implementation", "Status",
                    "Evidence", "Justification"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["Overall alignment conclusion", CONCLUSION, "", "", ""])

    allowed = {"Fulfilled", "Fulfilled with justified refinement",
               "Removed under proposal contingency", "Expanded beyond proposal"}
    bad = [r[2] for r in rows if r[2] not in allowed]
    if bad:
        raise SystemExit(f"invalid status values: {bad}")
    print(f"wrote {path} ({len(rows)} commitments)")
    for status in sorted(allowed):
        print(f"  {status}: {sum(1 for r in rows if r[2] == status)}")


if __name__ == "__main__":
    main()

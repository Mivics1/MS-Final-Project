"""
Score the credential-harvesting screen against the manual audit judgements.

The audit reviewed a seeded stratified random sample of 100 messages (25 from
each of retained/excluded x real-world/AI, drawn by audit_screen_sample.py).
DISAGREEMENTS below record every sampled message whose manually assigned label
differs from the screen's decision; every other sampled message was judged to
agree with the screen. Recording only the disagreements keeps the judgement set
short enough to check by hand.

Single-rater audit conducted by the project author; no second rater was
available, so no inter-rater agreement statistic is reported.

Estimates are computed per source, weighting each stratum by its population
size, so precision/recall generalise to the whole screened corpus rather than
to the sample.

Run:  PYTHONPATH=. .venv/bin/python scripts/audit_screen_score.py
Outputs: results/screen_audit.json

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

import json

# Sampled messages whose manual label CONTRADICTS the screen.
#   in a 'retained' stratum -> judged NOT credential-harvesting (false positive)
#   in an 'excluded' stratum -> judged credential-harvesting   (false negative)
DISAGREEMENTS = {
    # retained but judged not credential-harvesting (document/file-share,
    # payment-advice and legal-notice lures rather than credential capture)
    "nazario|retained|phishing-2024#144",
    "nazario|retained|phishing-2024#113",
    "nazario|retained|phishing-2023#261",
    "nazario|retained|phishing-2024#31",
    "nazario|retained|phishing-2024#254",
    # AI messages retained on a "confirm your ..." phrase that actually
    # concerns event or appointment attendance
    "ai|retained|email_494.txt",
    "ai|retained|email_504.txt",
    # excluded but judged credential-harvesting (account-termination,
    # mailbox-recovery, password-expiry and billing-update lures whose
    # call-to-action wording still falls outside the rule)
    "nazario|excluded|phishing-2022#79",
    "nazario|excluded|phishing-2023#296",
    "nazario|excluded|phishing-2023#337",
    "nazario|excluded|phishing-2022#0",
    "nazario|excluded|phishing-2022#236",
}

# Stratum population sizes reported by audit_screen_sample.py for the final
# corpus build (used to weight the sample-based rates).
POPULATION = {
    "nazario|retained": 652, "nazario|excluded": 167,
    "ai|retained": 559, "ai|excluded": 306,
}


def main() -> None:
    sample = json.load(open("results/screen_audit_sample.json"))
    per_stratum: dict[str, dict[str, int]] = {}
    for row in sample:
        stratum = f"{row['source']}|{'retained' if row['screen_retained'] else 'excluded'}"
        d = per_stratum.setdefault(stratum, {"n": 0, "agree": 0, "disagree": 0})
        d["n"] += 1
        if row["key"] in DISAGREEMENTS:
            d["disagree"] += 1
        else:
            d["agree"] += 1

    out = {"per_stratum": per_stratum, "population": POPULATION, "by_source": {}}
    for src in ("nazario", "ai"):
        ret, exc = per_stratum[f"{src}|retained"], per_stratum[f"{src}|excluded"]
        # precision of retention = share of retained judged truly in-scope
        precision = ret["agree"] / ret["n"]
        # false-negative rate among excluded = share judged truly in-scope
        fn_rate = exc["disagree"] / exc["n"]
        est_tp = precision * POPULATION[f"{src}|retained"]
        est_fn = fn_rate * POPULATION[f"{src}|excluded"]
        out["by_source"][src] = {
            "sampled_retained": ret["n"],
            "sampled_excluded": exc["n"],
            "precision_of_retained": round(precision, 3),
            "false_negative_rate_among_excluded": round(fn_rate, 3),
            "estimated_recall": round(est_tp / (est_tp + est_fn), 3) if est_tp + est_fn else None,
            "estimated_in_scope_retained": round(est_tp),
            "estimated_in_scope_missed": round(est_fn),
        }

    with open("results/screen_audit.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["by_source"], indent=2))


if __name__ == "__main__":
    main()

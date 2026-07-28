"""
Produce Dissertation_Final_Validated.docx from the authoritative aligned
baseline.

The aligned document is authoritative for wording, scope and interpretation, so
this script does NOT regenerate it. It copies the baseline and makes only
presentation changes that were requested for finalisation:

  * replaces the four figures with the freshly regenerated high-resolution
    renders, sized to the requested print widths;
  * preserves each image's true aspect ratio (height is derived from the new
    image's own proportions, never fixed independently of width);
  * keeps every paragraph, table, caption, heading, the populated table of
    contents and both lists exactly as the baseline has them.

Run:  PYTHONPATH=. .venv/bin/python scripts/finalise_validated_docx.py

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess

import docx
import docx.text.paragraph
from docx.shared import Cm
from PIL import Image

ART = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ART)
BASELINE = "/Users/daramola/Downloads/Dissertation_Final_Aligned_Final.docx"
OUT_DIR = os.path.join(REPO, "MS-Final-Dissertation")
OUT = os.path.join(OUT_DIR, "Dissertation_Final_Validated.docx")

# figure order in the document, with the requested print width
FIGURES = [
    ("Figure 4.1", os.path.join(ART, "results", "fig_architecture.png"), 16.0),
    ("Figure 6.1", os.path.join(ART, "results", "confusion_logreg.png"), 11.5),
    ("Figure 6.2", os.path.join(ART, "results", "feature_importance.png"), 15.5),
    ("Figure 6.3", os.path.join(ART, "results", "fig_readability.png"), 15.5),
]


RELEASE_TAG = "v1.3-validated"

# Appendix A must list the environment-setup and validation steps, and must
# name the script that actually produces this document.
APPENDIX_A_STEPS = [
    "python3 -m venv .venv and pip install -r requirements.txt — environment setup",
    "python scripts/build_corpus.py — rebuild data/emails.csv from the raw sources",
    "python -m src.pipeline --data data/emails.csv --out results/ — train and evaluate",
    "python scripts/supplementary_analysis.py — class means, importance and ablation",
    "python scripts/grouped_split_eval.py — similarity-cluster split and 2022 sensitivity",
    "python scripts/audit_screen_sample.py and audit_screen_score.py — screening audit",
    "python scripts/make_figures.py — architecture and readability figures",
    "python scripts/validate_final_alignment.py — verify the document against the outputs",
    "python scripts/finalise_validated_docx.py and scripts/repaginate_toc.py — assemble "
    "this document and refresh its contents pages",
]


def update_appendix_a(doc, commit: str) -> None:
    """Refresh the reproduction steps and the cited artefact state."""
    paras = doc.paragraphs
    heads = [i for i, p in enumerate(paras)
             if p.text.strip().startswith("Appendix A:")
             and "\t" not in p.text]                    # skip the contents entry
    if not heads:
        raise SystemExit("Appendix A heading not found")
    start = heads[-1]

    # 1. update the cited release and commit in the lead paragraph
    for p in paras[start:start + 4]:
        for run in p.runs:
            if "release" in run.text and "commit" in run.text:
                run.text = re.sub(
                    r"release \S+, commit \w+",
                    f"release {RELEASE_TAG}, commit {commit}", run.text)

    # 2. rewrite the numbered step list
    steps = [p for p in paras[start:start + 20]
             if re.match(r"^\d+\.\s+(python|pip)", p.text.strip())]
    if not steps:
        raise SystemExit("Appendix A step list not found")
    template = steps[0]
    for p, text in zip(steps, APPENDIX_A_STEPS):
        if p.runs:
            p.runs[0].text = text
            for r in p.runs[1:]:
                r.text = ""
    # append any steps beyond those already present, reusing the list style
    for text in APPENDIX_A_STEPS[len(steps):]:
        new_p = copy.deepcopy(template._p)
        template._p.addprevious(new_p) if False else steps[-1]._p.addnext(new_p)
        para = docx.text.paragraph.Paragraph(new_p, template._parent)
        if para.runs:
            para.runs[0].text = text
            for r in para.runs[1:]:
                r.text = ""
        steps.append(para)
    print(f"Appendix A: {len(APPENDIX_A_STEPS)} reproduction steps, "
          f"release {RELEASE_TAG} commit {commit}")


def current_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ART,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "(see repository release page)"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.copyfile(BASELINE, OUT)
    doc = docx.Document(OUT)

    shapes = doc.inline_shapes
    if len(shapes) != len(FIGURES):
        raise SystemExit(f"expected {len(FIGURES)} figures, found {len(shapes)}")

    for shape, (label, path, width_cm) in zip(shapes, FIGURES):
        if not os.path.exists(path):
            raise SystemExit(f"missing regenerated figure: {path}")
        # swap the image bytes in place, keeping the existing relationship
        rid = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        part = doc.part.related_parts[rid]
        with open(path, "rb") as fh:
            part._blob = fh.read()
        # width is chosen; height follows the new image's own aspect ratio
        with Image.open(path) as im:
            px_w, px_h = im.size
        shape.width = Cm(width_cm)
        shape.height = Cm(width_cm * px_h / px_w)
        print(f"{label}: {os.path.basename(path)} {px_w}x{px_h}px -> "
              f"{width_cm:.1f} x {width_cm * px_h / px_w:.2f} cm "
              f"({px_w / (width_cm / 2.54):.0f} dpi)")

    update_appendix_a(doc, current_commit())
    doc.save(OUT)
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()

"""
Correct the page numbers in the validated dissertation's table of contents.

The aligned baseline's table of contents is static text, not a Word TOC field
(it contains no fldChar, instrText or PAGEREF markers), so it cannot refresh
itself and went stale when the figures were resized. This script reads the true
page number of every heading from the exported PDF and rewrites the numbers in
place, leaving the entry titles, styles, tab stops and ordering untouched.

Because rewriting a number can itself reflow the document, the caller should
export the PDF again and re-run until the pagination is stable; run_until_stable
does that automatically.

Run:  PYTHONPATH=. .venv/bin/python scripts/repaginate_toc.py

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import docx
from pypdf import PdfReader

ART = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ART)
DOCX = os.path.join(REPO, "MS-Final-Dissertation", "Dissertation_Final_Validated.docx")
PDF = os.path.join(REPO, "MS-Final-Dissertation", "Dissertation_Final_Validated.pdf")
SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
MACRO = ("vnd.sun.star.script:Standard.Module1.UpdateAndExportPDF"
         "?language=Basic&location=application")


def export_pdf() -> None:
    subprocess.run(["pkill", "-f", "soffice"], capture_output=True)
    subprocess.run([SOFFICE, "--headless", "--norestore", MACRO],
                   capture_output=True, timeout=600)


def heading_pages() -> dict[str, int]:
    """Map heading text -> printed folio, from the rendered PDF."""
    reader = PdfReader(PDF)
    pages = [(p.extract_text() or "") for p in reader.pages]
    folios, bodies = [], []
    for text in pages:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        folio = int(lines[0]) if lines and re.fullmatch(r"\d{1,3}", lines[0]) else None
        folios.append(folio)
        bodies.append(lines[1:] if folio is not None else lines)

    found: dict[str, int] = {}
    for folio, lines in zip(folios, bodies):
        if folio is None:
            continue                      # preliminary pages use Roman numerals
        for line in lines:
            if re.match(r"^\d+(\.\d+)?\.?\s+\S|^References$|^Appendix [AB]:", line):
                found.setdefault(line.strip(), folio)
    return found


def normalise(title: str) -> str:
    """TOC titles and heading text differ only in whitespace and dash glyphs."""
    t = title.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", t).strip().lower()


def repaginate() -> int:
    """Rewrite stale TOC page numbers. Returns the number of entries changed."""
    doc = docx.Document(DOCX)
    pages = {normalise(k): v for k, v in heading_pages().items()}

    in_toc = False
    changed = 0
    for para in doc.paragraphs:
        text = para.text.strip()
        if text == "Table of Contents":
            in_toc = True
            continue
        if in_toc and text in ("List of Figures", "List of Tables"):
            break
        if not in_toc or "\t" not in para.text:
            continue

        title, _, current = para.text.rpartition("\t")
        if not current.strip().isdigit():
            continue
        key = normalise(title)
        # heading text in the body may carry a trailing section title only
        actual = pages.get(key)
        if actual is None:
            continue
        if int(current.strip()) != actual:
            new_text = f"{title}\t{actual}"
            # single-run entries: rewrite the run, preserving its formatting
            if len(para.runs) == 1:
                para.runs[0].text = new_text
            else:
                para.runs[0].text = new_text
                for r in para.runs[1:]:
                    r.text = ""
            changed += 1
            print(f"  {title.strip()[:50]:52} {current.strip()} -> {actual}")
    if changed:
        doc.save(DOCX)
    return changed


def run_until_stable(max_rounds: int = 4) -> None:
    for rnd in range(1, max_rounds + 1):
        print(f"[round {rnd}] exporting PDF and checking pagination")
        export_pdf()
        changed = repaginate()
        if changed == 0:
            print("pagination is stable; table of contents matches the document")
            export_pdf()
            return
        print(f"[round {rnd}] corrected {changed} entries")
    print("WARNING: pagination did not stabilise", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    run_until_stable()

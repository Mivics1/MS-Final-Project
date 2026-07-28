"""
Generate the two figures that are not produced by the main pipeline:
  results/fig_architecture.png  - pipeline architecture (Figure 4.1)
  results/fig_readability.png   - readability distribution by class (Figure 6.x)

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from src import data_loader, features


# Figure 4.1 is placed at 16.2 cm wide in Word. Rendering 6.38 in wide at
# 450 dpi gives ~2,870 px, comfortably above 300 dpi at final print size.
FIG41_WIDTH_IN = 6.38
FIG41_DPI = 450


def architecture() -> None:
    """Seven-stage horizontal pipeline diagram (Figure 4.1).

    Boxes are equal width, gaps are equal, and the figure is sized so that at
    16.2 cm in Word the labels remain readable at 100% zoom.
    """
    # Labels are wrapped so that no line exceeds the box width at the chosen
    # font size; the previous single-line "feature extraction" was clipped.
    stages = [
        ("1\nCorpus\nsources", "#264653"),
        ("2\nCleaning &\nscreening", "#2A6F73"),
        ("3\nDeduplication\n& splitting", "#287271"),
        ("4\nStylometric\nfeature\nextraction", "#4C956C"),
        ("5\nModel\ntraining", "#8AB17D"),
        ("6\nHeld-out\nevaluation", "#E9C46A"),
        ("7\nFeature\ninterpretation", "#E76F51"),
    ]
    n = len(stages)
    w, h, gap = 1.0, 0.94, 0.26
    total_w = n * w + (n - 1) * gap

    fig_h = FIG41_WIDTH_IN * (h + 0.16) / total_w
    fig, ax = plt.subplots(figsize=(FIG41_WIDTH_IN, fig_h))
    ax.axis("off")
    ax.set_aspect("equal")

    for i, (label, colour) in enumerate(stages):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch(
            (x, 0), w, h, boxstyle="round,pad=0.012,rounding_size=0.07",
            linewidth=0, facecolor=colour))
        ax.text(x + w / 2, h / 2, label, ha="center", va="center",
                fontsize=5.9, color="white", weight="bold", linespacing=1.3)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x + w + 0.045, h / 2), (x + w + gap - 0.045, h / 2),
                arrowstyle="-|>", mutation_scale=9, linewidth=1.0,
                color="#333333", shrinkA=0, shrinkB=0))

    ax.set_xlim(-0.05, total_w + 0.05)
    ax.set_ylim(-0.08, h + 0.08)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig("results/fig_architecture.png", dpi=FIG41_DPI,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def readability() -> None:
    df = data_loader.load_csv("data/emails.csv")
    names = features.FEATURE_NAMES
    X = features.extract_matrix(df["text"].tolist())
    y = df["label"].to_numpy()
    idx = names.index("flesch_reading_ease")
    human = X[y == 0, idx]
    llm = X[y == 1, idx]
    # placed at ~15.5 cm in Word; 6.1 in at 400 dpi = ~2,440 px
    fig, ax = plt.subplots(figsize=(6.1, 3.3))
    bins = np.linspace(-20, 120, 40)
    hm, lm = float(np.mean(human)), float(np.mean(llm))
    ax.hist(human, bins=bins, alpha=0.65, label="Real-world (Nazario)", color="#2A6F73")
    ax.hist(llm, bins=bins, alpha=0.65, label="AI-generated", color="#E76F51")
    ax.axvline(hm, color="#2A6F73", linestyle="--", linewidth=1.4)
    ax.axvline(lm, color="#E76F51", linestyle="--", linewidth=1.4)
    ax.annotate(f"mean {hm:.1f}", xy=(hm, ax.get_ylim()[1] * 0.92),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8, color="#2A6F73", weight="bold")
    ax.annotate(f"mean {lm:.1f}", xy=(lm, ax.get_ylim()[1] * 0.92),
                xytext=(-52, 0), textcoords="offset points",
                fontsize=8, color="#E76F51", weight="bold")
    ax.set_xlabel("Flesch Reading Ease (higher = easier to read)", fontsize=9)
    ax.set_ylabel("Number of emails", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig("results/fig_readability.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    readability()
    print("wrote results/fig_architecture.png and results/fig_readability.png")

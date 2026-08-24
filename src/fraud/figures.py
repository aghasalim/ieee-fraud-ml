"""Draw the README figures from reports/*.csv.

Reads the saved reports only -- no data download, no training, no credentials.
Every number is the one already quoted in the README tables.

    python -m fraud.figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def leakage(out: Path) -> Path:
    """What the evaluation protocol alone is worth in AUC.

    Nothing about the model changes across these six bars. Only how the folds are
    cut and whether target encoding is fitted globally or per fold. The spread is
    10.4 AUC points, which is larger than any modelling gain in this repository.
    """
    table = pd.read_csv(REPORTS / "leakage_real.csv")
    splits = list(dict.fromkeys(table["split"]))
    encodings = list(dict.fromkeys(table["target encoding"]))
    base = np.arange(len(splits))

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [2, 1]}
    )
    for offset, encoding in enumerate(encodings):
        rows = table[table["target encoding"] == encoding].set_index("split").loc[splits]
        left.bar(base + (offset - 0.5) * 0.36, rows["AUC"], 0.36,
                 label=f"{encoding} target encoding",
                 color="#b2182b" if encoding == "global" else "#2166ac",
                 edgecolor="0.3", lw=0.5)
        for index, value in enumerate(rows["AUC"]):
            left.text(index + (offset - 0.5) * 0.36, value + 0.004, f"{value:.3f}",
                      ha="center", fontsize=8)
    left.set_xticks(base)
    left.set_xticklabels(splits, fontsize=9)
    left.set_ylim(0.80, 1.0)
    left.set_ylabel("AUC")
    left.set_title(
        "Same model, same features, same rows.\nOnly the evaluation protocol changes.",
        fontsize=10,
    )
    left.legend(frameon=False, fontsize=8)
    left.spines[["top", "right"]].set_visible(False)

    overlap = table.drop_duplicates("split").set_index("split").loc[splits]
    right.bar(base, overlap["card overlap (fold 1)"] * 100, 0.5,
              color="#f4a582", edgecolor="0.3", lw=0.5)
    right.set_xticks(base)
    right.set_xticklabels(splits, rotation=20, ha="right", fontsize=8)
    right.set_ylabel("% of validation cards seen in training")
    right.set_title("the mechanism: card overlap", fontsize=10)
    right.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def ablation(out: Path) -> Path:
    """Feature groups added one at a time, against the train-validation gap.

    Target encoding is the instructive row: it drives training AUC to exactly
    1.000 and costs 3.1 points of validation AUC. A leaderboard that reported
    training performance would rank it first.
    """
    table = pd.read_csv(REPORTS / "ablation.csv")
    positions = np.arange(len(table))

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    colours = ["#b2182b" if "target encoding" in f else "#2166ac"
               for f in table["features"]]
    left.barh(positions, table["val AUC"], color=colours, edgecolor="0.3", lw=0.5)
    left.set_yticks(positions)
    left.set_yticklabels(table["features"], fontsize=9)
    left.invert_yaxis()
    left.set_xlim(0.84, 0.89)
    left.set_xlabel("validation AUC")
    left.set_title("validation AUC (what you keep)", fontsize=10)
    left.spines[["top", "right"]].set_visible(False)

    right.barh(positions, table["train-val gap"], color=colours,
               edgecolor="0.3", lw=0.5)
    right.set_xlabel("train − validation AUC gap")
    right.set_title("the gap (what you were fooled by)", fontsize=10)
    right.spines[["top", "right"]].set_visible(False)
    for index, (_, row) in enumerate(table.iterrows()):
        right.text(row["train-val gap"] + 0.002, index,
                   f"train {row['train AUC']:.3f}", va="center", fontsize=7.5,
                   color="0.35")

    figure.suptitle(
        "Adding target encoding takes training AUC to 1.000 and costs 3.1 points "
        "of validation AUC.",
        fontsize=10, y=0.02, color="0.35",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def calibration(out: Path) -> Path:
    """Reliability of the predicted probabilities, which ranking metrics ignore."""
    table = pd.read_csv(REPORTS / "ea_calibration.csv")

    figure, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1.2, label="perfect calibration")
    ax.plot(table["mean_pred"], table["actual"], "o-", color="#2166ac", lw=2,
            markersize=7, label="model")
    for _, row in table.iterrows():
        ax.annotate(f"n={int(row['n']):,}", (row["mean_pred"], row["actual"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=7,
                    color="0.4")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraud rate")
    ax.set_title(
        "Under-confident in the low buckets, where almost all the volume is.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def review_budget(out: Path) -> Path:
    """Recall and precision against how many transactions a team can review.

    AUC is a summary over every threshold, including ones nobody would operate at.
    A review team has a fixed daily capacity, and this is the curve they live on.
    """
    table = pd.read_csv(REPORTS / "ea_budget.csv")
    positions = np.arange(len(table))

    figure, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(positions, table["recall"] * 100, "o-", color="#2166ac", lw=2,
            label="fraud recall")
    ax.plot(positions, table["precision"] * 100, "s-", color="#b2182b", lw=2,
            label="precision")
    for index, row in table.iterrows():
        ax.annotate(f"{row['recall'] * 100:.0f}%",
                    (index, row["recall"] * 100), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="#2166ac")
    ax.set_xticks(positions)
    ax.set_xticklabels(table["alert budget"])
    ax.set_xlabel("alert budget (share of transactions reviewed)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title(
        "The operating curve a review team actually sits on, "
        "which a single AUC hides.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def segments(out: Path) -> Path:
    """Per-segment performance, because a global AUC averages over the weak ones."""
    table = pd.read_csv(REPORTS / "ea_segments.csv").copy()
    table["label"] = table["segment"].astype(str) + "=" + table["value"].astype(str)
    table = table.sort_values("AUC")
    positions = np.arange(len(table))

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    left.barh(positions, table["AUC"], color="#2166ac", edgecolor="0.3", lw=0.5)
    left.set_yticks(positions)
    left.set_yticklabels(table["label"], fontsize=8)
    left.set_xlim(0.6, max(table["AUC"]) * 1.02)
    left.set_xlabel("AUC within segment")
    left.set_title("segment AUC", fontsize=10)
    left.spines[["top", "right"]].set_visible(False)

    right.barh(positions, table["recall@1%"] * 100, color="#f4a582",
               edgecolor="0.3", lw=0.5)
    right.set_xlabel("recall at a 1% alert budget (%)")
    right.set_title("what that means at a fixed budget", fontsize=10)
    right.spines[["top", "right"]].set_visible(False)
    for index, (_, row) in enumerate(table.iterrows()):
        right.text(row["recall@1%"] * 100 + 0.4, index,
                   f"n={int(row['n']):,}  fraud {row['fraud rate'] * 100:.1f}%",
                   va="center", fontsize=7, color="0.4")

    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        leakage(FIGURES / "leakage.png"),
        ablation(FIGURES / "ablation.png"),
        calibration(FIGURES / "calibration.png"),
        review_budget(FIGURES / "review-budget.png"),
        segments(FIGURES / "segments.png"),
    ):
        print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

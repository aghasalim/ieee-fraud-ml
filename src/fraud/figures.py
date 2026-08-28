"""Draw the README figures from reports/*.csv.

Reads the saved reports only, no data download, no training, no credentials.
Every number here is one already quoted in a README table.

    python -m src.fraud.figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Red is the leaky or costly variant everywhere in the README prose, blue is the
# honest one. Keep that pairing fixed across figures.
LEAKY = PALETTE[1]
HONEST = PALETTE[0]
MECHANISM = PALETTE[3]


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
    wrapped = [s.replace(" + ", "\n+ ") for s in splits]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [2, 1]}
    )
    for offset, encoding in enumerate(encodings):
        rows = table[table["target encoding"] == encoding].set_index("split").loc[splits]
        left.bar(base + (offset - 0.5) * 0.36, rows["AUC"], 0.36,
                 label=f"{encoding} target encoding",
                 color=LEAKY if encoding == "global" else HONEST)
        for index, value in enumerate(rows["AUC"]):
            left.text(index + (offset - 0.5) * 0.36, value + 0.004, f"{value:.3f}",
                      ha="center", fontsize=9)
    left.set_xticks(base)
    left.set_xticklabels(splits)
    left.set_ylim(0.80, 1.02)
    left.set_ylabel("AUC (0 to 1)")
    left.set_xlabel("how the folds are cut")
    left.legend(loc="upper right", borderaxespad=0.6)
    titled(left, "Changing only the protocol moves AUC by 10.4 points",
           "Same LightGBM, same 432 features, same 590,540 rows in all six bars.")

    overlap = table.drop_duplicates("split").set_index("split").loc[splits]
    right.bar(base, overlap["card overlap (fold 1)"] * 100, 0.55, color=MECHANISM)
    for index, value in enumerate(overlap["card overlap (fold 1)"] * 100):
        right.text(index, value + 1.5, f"{value:.0f}%", ha="center", fontsize=9)
    right.set_xticks(base)
    right.set_xticklabels(wrapped)
    right.set_ylim(0, 100)
    right.set_ylabel("validation cards also seen in training (%)")
    titled(right, "Card overlap falls with it",
           "The mechanism, measured on fold 1.")

    figure.tight_layout()
    figure.savefig(out)
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
    colours = [LEAKY if "target encoding" in f else HONEST for f in table["features"]]

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)

    # Dots, not bars: the axis starts at 0.845, and a bar whose baseline is not
    # zero makes the length of the bar mean nothing.
    left.scatter(table["val AUC"], positions, s=110, color=colours, zorder=3)
    for index, value in enumerate(table["val AUC"]):
        left.text(value + 0.0012, index, f"{value:.4f}", va="center", fontsize=9,
                  color="#444444")
    left.set_yticks(positions)
    left.set_yticklabels(table["features"])
    left.invert_yaxis()
    left.set_xlim(0.845, 0.893)
    left.set_xlabel("validation AUC (0 to 1)")
    titled(left, "Only target encoding makes validation worse",
           "Feature groups added cumulatively, chronological folds, 30 day embargo.")

    right.barh(positions, table["train-val gap"], 0.62, color=colours)
    for index, (_, row) in enumerate(table.iterrows()):
        right.text(row["train-val gap"] + 0.003, index,
                   f"train {row['train AUC']:.4f}", va="center", fontsize=9,
                   color="#444444")
    right.set_xlim(0, 0.20)
    right.set_xlabel("train AUC minus validation AUC (points)")
    titled(right, "It pays for that by memorising cards",
           "13,553 cards, so the encoding is close to a unique key per customer.")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def calibration(out: Path) -> Path:
    """Reliability of the predicted probabilities, which ranking metrics ignore."""
    table = pd.read_csv(REPORTS / "ea_calibration.csv")
    rows = int(table["n"].sum())

    # Log axes because six of the seven buckets sit under 0.4 and the whole point
    # of the figure is the bucket at 0.002.
    figure, ax = plt.subplots(figsize=(8.0, 5.6))
    line = np.array([0.0015, 1.05])
    ax.plot(line, line, ls="--", color="#8c8c8c", lw=1.2, zorder=1,
            label="perfect calibration")
    ax.vlines(table["mean_pred"], table["mean_pred"], table["actual"],
              color="#b9b9b9", lw=1.0, zorder=1)
    ax.plot(table["mean_pred"], table["actual"], "o-", color=HONEST, zorder=3,
            markersize=7, label="this model")
    for index, (_, row) in enumerate(table.iterrows()):
        # Above-left of the marker, which is the empty side of the curve. The
        # first bucket is against the y axis, so that one goes below-right.
        offset, align = ((10, -15), "left") if index == 0 else ((-9, 8), "right")
        ax.annotate(f"n={int(row['n']):,}", (row["mean_pred"], row["actual"]),
                    textcoords="offset points", xytext=offset, ha=align,
                    fontsize=8.5, color="#5a5a5a")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.0015, 1.6)
    ax.set_ylim(0.010, 1.6)
    ax.set_xlabel("mean predicted probability of fraud (0 to 1, log scale)")
    ax.set_ylabel("observed fraud rate in the bucket (0 to 1, log scale)")
    ax.legend(loc="lower right", borderaxespad=1.0)
    titled(ax, "Under 1% the model predicts fraud about 7 times too low",
           f"Seven score buckets over {rows:,} out-of-fold rows. "
           "Grey stems are the miss.")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def review_budget(out: Path) -> Path:
    """Recall and precision against how many transactions a team can review.

    AUC is a summary over every threshold, including ones nobody would operate at.
    A review team has a fixed daily capacity, and this is the curve they live on.
    """
    table = pd.read_csv(REPORTS / "ea_budget.csv")
    positions = np.arange(len(table))

    figure, ax = plt.subplots(figsize=(9.5, 5.0))
    series = [("recall", "fraud caught (recall)", HONEST, "o", 12),
              ("precision", "precision of the queue", LEAKY, "s", -16)]
    for column, name, colour, marker, dy in series:
        values = table[column] * 100
        ax.plot(positions, values, marker=marker, color=colour, zorder=3)
        for index, value in enumerate(values):
            ax.annotate(f"{value:.0f}%", (index, value), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=9, color=colour)
        # Labelled at the end of the line instead of in a legend box, so nothing
        # is drawn on top of the data.
        ax.text(positions[-1] + 0.12, values.iloc[-1], name, va="center",
                fontsize=10, color=colour)

    ax.set_xticks(positions)
    ax.set_xticklabels(table["alert budget"])
    ax.set_xlim(-0.25, len(table) + 0.05)
    ax.set_ylim(-8, 118)
    ax.set_xlabel("alert budget (share of transactions sent to review, %)")
    ax.set_ylabel("percent (%)")
    titled(ax, "A 1% review queue catches 24% of fraud at 89% precision",
           "Validation rows ranked by score, top slice reviewed. Four measured budgets.")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


# ea_segments.csv stores ProductCD as the integer pandas hands out in
# `train.prepare()`: `.astype("category").cat.codes` numbers the categories in
# sorted order, so 0..4 are C, H, R, S, W. Verified against the raw column:
# code 4 is W with n=355,414, the row the README quotes.
PRODUCT_CD = ["C", "H", "R", "S", "W"]


def segments(out: Path) -> Path:
    """Per-segment performance, because a global AUC averages over the weak ones."""
    table = pd.read_csv(REPORTS / "ea_segments.csv").copy()
    table["label"] = [
        f"{seg}=" + (PRODUCT_CD[int(val)] if seg == "ProductCD" else str(val))
        for seg, val in zip(table["segment"], table["value"])
    ]
    table = table.sort_values("AUC")
    positions = np.arange(len(table))
    colours = [LEAKY if n > 300_000 else HONEST for n in table["n"]]

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 6.6), sharey=True)

    # Dots again: the axis starts at 0.68, so bar length would be meaningless.
    left.scatter(table["AUC"], positions, s=95, color=colours, zorder=3)
    for index, value in enumerate(table["AUC"]):
        left.text(value + 0.004, index, f"{value:.3f}", va="center", fontsize=8.5,
                  color="#444444")
    left.set_yticks(positions)
    left.set_yticklabels(table["label"])
    left.set_xlim(0.68, 0.92)
    left.set_xlabel("AUC inside the segment (0 to 1)")
    titled(left, "The two largest segments are the two it reads worst",
           "Red marks the two segments holding over 300,000 rows each.")

    right.barh(positions, table["recall@1%"] * 100, 0.62, color=colours)
    for index, (_, row) in enumerate(table.iterrows()):
        right.text(row["recall@1%"] * 100 + 0.7, index,
                   f"n={int(row['n']):,}   fraud {row['fraud rate'] * 100:.1f}%",
                   va="center", fontsize=8.5, color="#5a5a5a")
    right.set_xlim(0, 52)
    right.set_xlabel("segment fraud found in its own top 1% of scores (%)")
    titled(right, "At a 1% budget those two find 14% of their own fraud",
           "Same segments, same order. The best slices reach twice that.")

    figure.tight_layout()
    figure.savefig(out)
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

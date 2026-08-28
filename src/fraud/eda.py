"""EDA on the real competition data, the numbers behind the README's
"What the data actually looks like" table.

This module was referenced by `make eda` and by the README's feature checklist
before it existed, so the target failed with ModuleNotFoundError and none of
those figures had committed code behind them. The numbers themselves were
correct; there was simply no way for a reader to regenerate them.

Everything here is computed on the raw transaction table, before any join or
feature engineering, so the figures describe the data as downloaded rather than
as transformed.
"""
from __future__ import annotations

import pandas as pd

from . import config

# Missingness band used in the README. The interesting columns are the ones
# missing enough to matter but not so much as to be useless: a column that is
# 99% empty is an easy drop decision, one that is 70% empty is not.
MISS_LO, MISS_HI = 0.5, 0.9


def compute() -> dict[str, pd.DataFrame]:
    tx = pd.read_csv(config.RAW / "train_transaction.csv")
    idf = pd.read_csv(config.RAW / "train_identity.csv")

    miss = tx.isnull().mean()
    in_band = ((miss >= MISS_LO) & (miss <= MISS_HI)).sum()
    coverage = tx[config.ID_COL].isin(idf[config.ID_COL]).mean()
    span_days = (tx[config.TIME_COL].max() - tx[config.TIME_COL].min()) / config.DAY

    # The gap to the test period is the single most consequential fact about this
    # dataset: it is why contiguous chronological folds are still too easy, and
    # why the honest configuration uses a 30-day embargo.
    gap_days = float("nan")
    test_path = config.RAW / "test_transaction.csv"
    if test_path.exists():
        te = pd.read_csv(test_path, usecols=[config.TIME_COL])
        gap_days = (te[config.TIME_COL].min() - tx[config.TIME_COL].max()) / config.DAY

    by_product = (
        tx.groupby("ProductCD")[config.TARGET]
        .agg(n="size", fraud_rate="mean")
        .sort_values("fraud_rate")
        .reset_index()
    )

    summary = pd.DataFrame(
        [
            ("transactions", f"{tx.shape[0]} x {tx.shape[1]}"),
            ("identity rows", f"{idf.shape[0]}"),
            ("fraud rate", f"{tx[config.TARGET].mean():.5f}"),
            ("identity coverage", f"{coverage:.4f}"),
            (f"columns {MISS_LO:.0%}-{MISS_HI:.0%} missing", f"{in_band}"),
            ("worst column missingness", f"{miss.max():.4f}"),
            ("fraud-rate spread across ProductCD",
             f"{by_product.fraud_rate.max() / by_product.fraud_rate.min():.2f}x"),
            ("train span (days)", f"{span_days:.1f}"),
            ("gap to test period (days)", f"{gap_days:.1f}"),
        ],
        columns=["metric", "value"],
    )

    top = miss[miss > 0].sort_values(ascending=False).head(30)
    missing_top = pd.DataFrame({"column": top.index, "missing_frac": top.values})
    return {"eda_summary": summary, "eda_product": by_product, "eda_missing": missing_top}


def main() -> None:
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    for name, df in compute().items():
        out = config.REPORTS / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"\n{name}:")
        print(df.to_string(index=False))
        print(f"-> {out}")


if __name__ == "__main__":
    main()

"""Feature builders.

Every builder that summarises other rows takes a `tr_idx` and is fitted on those
rows only. That includes frequency encodings, which use no labels at all: they
still summarise *which transactions exist*, and at deployment you would only
know the past. Computing them over the full frame is a smaller sin than a global
target encoding but it is the same kind of sin, and entry 4 of NOTES.md is about
what happens when you assume the small version does not matter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

SMOOTH = 20.0


def base(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms of a single row, no cross-row information, so no fold needed."""
    out = df.copy()
    t = out[config.TIME_COL]
    out["hour"] = (t / 3600) % 24
    out["dayofweek"] = (t / config.DAY) % 7
    out["day_index"] = t / config.DAY  # position in the 182-day window
    amt = out["TransactionAmt"]
    out["amt_log"] = np.log1p(amt)
    # The fractional part of the amount. Round numbers behave differently from
    # amounts carrying cents, and foreign-currency conversions leave a
    # distinctive long tail here.
    out["amt_cents"] = ((amt - np.floor(amt)) * 100).round().astype("float32")
    out["amt_is_round"] = (out["amt_cents"] == 0).astype("int8")
    return out


def uid(df: pd.DataFrame) -> pd.Series:
    """A pseudo-customer id.

    card1 alone splits one person's cards apart; adding addr1 and the D1 offset
    (days since that card's first activity, so it is constant per card rather
    than drifting with the calendar) gets closer to one row per person.
    """
    d1n = (df["day_index"] - df["D1"]).round()
    return (df["card1"].astype("string") + "_" + df["addr1"].astype("string")
            + "_" + d1n.astype("string"))


def add_frequency(df: pd.DataFrame, tr_idx: np.ndarray, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        counts = out.iloc[tr_idx][c].value_counts()
        out[f"{c}_freq"] = out[c].map(counts).fillna(0).astype("float32")
    return out


def add_uid_aggs(df: pd.DataFrame, tr_idx: np.ndarray) -> pd.DataFrame:
    """How unusual is this transaction for this pseudo-customer?"""
    out = df.copy()
    u = out["_uid"]
    g = out.iloc[tr_idx].groupby(u.iloc[tr_idx])["TransactionAmt"]
    stats = g.agg(["mean", "std", "count"])
    out["uid_amt_mean"] = u.map(stats["mean"]).astype("float32")
    out["uid_amt_std"] = u.map(stats["std"]).astype("float32")
    out["uid_count"] = u.map(stats["count"]).fillna(0).astype("float32")
    # Deviation from the customer's own norm, which is the actual signal --
    # the raw mean mostly re-encodes what card1 already says.
    out["uid_amt_dev"] = (
        (out["TransactionAmt"] - out["uid_amt_mean"]) / (out["uid_amt_std"] + 1)
    ).astype("float32")
    return out


def add_target_encoding(df: pd.DataFrame, tr_idx: np.ndarray,
                        cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    prior = float(out.iloc[tr_idx][config.TARGET].mean())
    for c in cols:
        g = out.iloc[tr_idx].groupby(c)[config.TARGET].agg(["sum", "count"])
        m = (g["sum"] + prior * SMOOTH) / (g["count"] + SMOOTH)
        out[f"{c}_te"] = out[c].map(m).fillna(prior).astype("float32")
    return out

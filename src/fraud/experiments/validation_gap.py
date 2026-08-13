"""The headline experiment: how much does a wrong validation split flatter you?

Runs a 2x2. Two split strategies (shuffled K-fold vs chronological) crossed with
two ways of building a per-card target encoding (computed over the whole dataset
vs computed inside each fold). The cross matters because "leakage" in this
dataset has two independent sources and they are usually discussed as one thing:

  - the split lets the model see the future
  - the *feature* carries labels from the validation rows, whatever the split is

Only the diagonal is honest. Reporting all four is what makes the size of each
effect visible instead of asserted.

Data
----
Falls back to a synthetic generator when the Kaggle data is absent, so the
experiment is runnable by anyone. The synthetic set is built to have the two
properties that make IEEE-CIS awkward -- recurring card entities with persistent
latent risk, and a base rate that drifts over time -- and nothing else. Numbers
from it demonstrate the *mechanism*; they are not results about fraud, and the
README labels them as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .. import config, split

NOISE_COLS = ["f0", "f1", "f2"]
FEATURES = ["amount", "hour", "card_te", *NOISE_COLS]


def make_synthetic(n: int = 60_000, n_cards: int = 4_000, seed: int = config.SEED):
    """Recurring cards with persistent latent risk, plus base-rate drift.

    ~15 transactions per card: enough for a card's history to be genuinely
    informative, few enough that a target encoding which includes a row's own
    label is meaningfully contaminated.
    """
    rng = np.random.default_rng(seed)
    card = rng.integers(0, n_cards, n)
    card_logit = rng.normal(0, 1.2, n_cards)[card]  # persistent per-card risk

    t = np.arange(n)
    drift = 1.5 * (t / n)  # base rate rises over the period

    amount = rng.lognormal(3.0, 1.0, n)
    z_amount = (np.log(amount) - np.log(amount).mean()) / np.log(amount).std()
    hour = rng.integers(0, 24, n)

    # Intercept set so the positive rate lands near IEEE-CIS's ~3.5%. Severe
    # imbalance is not cosmetic here: it changes how much a target encoding can
    # memorise from a handful of transactions per card.
    logit = -4.85 + card_logit + 0.55 * z_amount + drift + 0.4 * (hour < 5)
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    return pd.DataFrame({
        config.TIME_COL: t * 100,
        "card": card,
        "amount": amount,
        "hour": hour,
        "y": y,
        **{c: rng.normal(size=n) for c in NOISE_COLS},
    })


def _te_map(df: pd.DataFrame, idx: np.ndarray, prior: float, m: float = 20.0):
    """Smoothed per-card target mean, fitted on `idx` only."""
    g = df.iloc[idx].groupby("card")["y"].agg(["sum", "count"])
    return (g["sum"] + prior * m) / (g["count"] + m)


def _apply_te(df: pd.DataFrame, mapping: pd.Series, prior: float) -> np.ndarray:
    return df["card"].map(mapping).fillna(prior).to_numpy()


def evaluate(df: pd.DataFrame, folds, global_te: bool) -> float:
    """Mean out-of-fold AUC under one (split, encoding) combination."""
    import lightgbm as lgb

    prior = float(df["y"].mean())
    global_map = _te_map(df, np.arange(len(df)), prior) if global_te else None

    aucs = []
    for tr, va in folds:
        d = df.copy()
        mapping = global_map if global_te else _te_map(df, tr, prior)
        d["card_te"] = _apply_te(d, mapping, prior)

        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=config.SEED, verbose=-1,
        )
        model.fit(d.iloc[tr][FEATURES], d.iloc[tr]["y"])
        p = model.predict_proba(d.iloc[va][FEATURES])[:, 1]
        aucs.append(roc_auc_score(d.iloc[va]["y"], p))
    return float(np.mean(aucs))


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = make_synthetic()
        source = "synthetic"
    else:
        source = "IEEE-CIS"
    print(f"data: {source}  rows={len(df):,}  positives={df['y'].mean():.2%}")

    shuffled = split.random_kfold(df)
    chrono = split.expanding_window_folds(df)

    rows = []
    for split_name, folds in (("shuffled K-fold", shuffled), ("chronological", chrono)):
        for te_name, gte in (("global", True), ("fold-local", False)):
            auc = evaluate(df, folds, gte)
            leak = split.max_train_time_beyond_val(df, *folds[0][:2])
            overlap = split.entity_overlap(df, folds[0][0], folds[0][1], "card")
            rows.append({
                "split": split_name,
                "target encoding": te_name,
                "AUC": round(auc, 4),
                "trains on future": leak > 0,
                "card overlap (fold 1)": round(overlap, 3),
            })
            print(f"  {split_name:16s} + {te_name:10s} TE -> AUC {auc:.4f}")

    out = pd.DataFrame(rows)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    dest = config.REPORTS / f"validation_gap_{source}.csv"
    out.to_csv(dest, index=False)

    honest = out.query("split == 'chronological' and `target encoding` == 'fold-local'")["AUC"].iloc[0]
    worst = out["AUC"].max()
    print(f"\nmost flattering setup reads {worst:.4f}; the defensible one reads {honest:.4f}")
    print(f"inflation: +{worst - honest:.4f} AUC of pure self-congratulation")
    print(f"-> {dest}")
    return out


if __name__ == "__main__":
    run()

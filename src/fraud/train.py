"""Incremental feature ablation under the honest split, plus overfitting checks.

Scored with chronological folds and a 30-day embargo -- the setting entry 5 of
NOTES.md argues is the only one comparable to the real train->test gap. Numbers
here are therefore *lower* than anything produced with contiguous folds, and
much lower than a shuffled split. That is the point.

Feature groups are added cumulatively and each delta is recorded, including the
negative ones. A group that does not earn its place is a result, not a mistake
to quietly drop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, data, features, split

N_FOLDS = 3
GAP = 30 * config.DAY
N_ESTIMATORS = 400
FREQ_COLS = ["card1", "card2", "addr1", "P_emaildomain", "_uid"]
TE_COLS = ["card1", "_uid"]
DROP = {config.ID_COL, config.TARGET, config.TIME_COL, "_uid"}


def prepare() -> pd.DataFrame:
    df = data.load_raw("train")
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("category").cat.codes.astype("int32")
    df = features.base(df)
    df["_uid"] = features.uid(df)
    return df


def _fit(df: pd.DataFrame, tr, va, cols) -> tuple[float, float]:
    """Returns (train AUC, validation AUC).

    Fixed n_estimators, deliberately: early stopping on the fold being scored
    selects the iteration count using the validation labels, which quietly turns
    the score into a best-case rather than an estimate. `overfit_report`
    measures how much that is worth.
    """
    import lightgbm as lgb

    y = df[config.TARGET].to_numpy()
    m = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS, learning_rate=0.05, num_leaves=63,
        colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
        min_child_samples=50, random_state=config.SEED, verbose=-1, n_jobs=-1,
    )
    m.fit(df.iloc[tr][cols], y[tr])
    tr_auc = roc_auc_score(y[tr], m.predict_proba(df.iloc[tr][cols])[:, 1])
    va_auc = roc_auc_score(y[va], m.predict_proba(df.iloc[va][cols])[:, 1])
    return float(tr_auc), float(va_auc)


def _build(df, tr_idx, groups: set[str]) -> tuple[pd.DataFrame, list[str]]:
    d = df
    if "frequency" in groups:
        d = features.add_frequency(d, tr_idx, FREQ_COLS)
    if "uid_aggs" in groups:
        d = features.add_uid_aggs(d, tr_idx)
    if "target_enc" in groups:
        d = features.add_target_encoding(d, tr_idx, TE_COLS)
    cols = [c for c in d.columns if c not in DROP]
    if "engineered_base" not in groups:
        cols = [c for c in cols
                if c not in {"hour", "dayofweek", "day_index", "amt_log",
                             "amt_cents", "amt_is_round"}]
    return d, cols


def ablation() -> pd.DataFrame:
    df = prepare()
    folds = split.expanding_window_folds(df, n_folds=N_FOLDS, gap=GAP)
    print(f"rows={len(df):,}  folds={len(folds)}  embargo={GAP/config.DAY:.0f}d")

    steps = [
        ("raw columns only", set()),
        ("+ engineered base", {"engineered_base"}),
        ("+ frequency encoding", {"engineered_base", "frequency"}),
        ("+ uid aggregates", {"engineered_base", "frequency", "uid_aggs"}),
        ("+ target encoding", {"engineered_base", "frequency", "uid_aggs", "target_enc"}),
    ]

    rows, prev = [], None
    for name, groups in steps:
        tr_aucs, va_aucs = [], []
        for tr, va in folds:
            d, cols = _build(df, tr, groups)
            a, b = _fit(d, tr, va, cols)
            tr_aucs.append(a)
            va_aucs.append(b)
        va = float(np.mean(va_aucs))
        rows.append({
            "features": name,
            "n_features": len(cols),
            "train AUC": round(float(np.mean(tr_aucs)), 4),
            "val AUC": round(va, 4),
            "delta": None if prev is None else round(va - prev, 4),
            "train-val gap": round(float(np.mean(tr_aucs)) - va, 4),
        })
        print(f"  {name:24s} val {va:.4f}"
              + ("" if prev is None else f"  ({va - prev:+.4f})"))
        prev = va

    out = pd.DataFrame(rows)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.REPORTS / "ablation.csv", index=False)
    print()
    print(out.to_string(index=False))
    return out


def overfit_report() -> pd.DataFrame:
    """What early stopping on the scored fold is worth -- i.e. how much you
    flatter yourself by tuning the iteration count on your own validation set."""
    import lightgbm as lgb

    df = prepare()
    folds = split.expanding_window_folds(df, n_folds=N_FOLDS, gap=GAP)
    y = df[config.TARGET].to_numpy()
    rows = []
    for i, (tr, va) in enumerate(folds, 1):
        d, cols = _build(df, tr, {"engineered_base", "frequency", "uid_aggs"})
        m = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.05, num_leaves=63,
            colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
            min_child_samples=50, random_state=config.SEED, verbose=-1, n_jobs=-1,
        )
        m.fit(d.iloc[tr][cols], y[tr], eval_set=[(d.iloc[va][cols], y[va])],
              eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
        best_iter = m.best_iteration_ or N_ESTIMATORS
        es_auc = roc_auc_score(y[va], m.predict_proba(d.iloc[va][cols])[:, 1])
        fixed_tr, fixed_va = _fit(d, tr, va, cols)
        rows.append({
            "fold": i,
            "best_iter (chosen on val)": best_iter,
            "AUC w/ early stopping": round(es_auc, 4),
            "AUC w/ fixed 400": round(fixed_va, 4),
            "early-stopping bonus": round(es_auc - fixed_va, 4),
            "train AUC (fixed)": round(fixed_tr, 4),
            "train-val gap": round(fixed_tr - fixed_va, 4),
        })
        print(f"  fold {i}: es {es_auc:.4f} vs fixed {fixed_va:.4f} "
              f"(+{es_auc - fixed_va:.4f}), best_iter={best_iter}")

    out = pd.DataFrame(rows)
    out.to_csv(config.REPORTS / "overfit.csv", index=False)
    print()
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    ablation()
    print("\n=== early stopping on the scored fold ===")
    overfit_report()

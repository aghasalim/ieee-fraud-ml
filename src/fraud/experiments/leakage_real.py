"""The 2x2 leakage experiment, on the real 590k-transaction dataset.

Same design as the synthetic version so the two are directly comparable: two
split strategies x two ways of computing a per-card target encoding. The
question this answers is not "does leakage exist", the synthetic run settled
that, but whether the *ordering* found there survives contact with real data,
where the entity is a genuine card identifier and the features are the
competition's own.

Estimand throughout is out-of-fold AUC, the competition metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .. import config, data, split

ENTITY = "card1"  # the closest thing to a stable customer id in this schema
N_FOLDS = 3       # fewer than the synthetic run: 434 columns x 590k rows x 12 fits
SMOOTH = 20.0
DROP = {config.ID_COL, config.TARGET, config.TIME_COL}


def prepare() -> pd.DataFrame:
    df = data.load_raw("train")
    # Label-encode object columns. No target involved, so this is a dtype
    # conversion rather than an encoding that could leak.
    for c in df.columns:
        if df[c].dtype == "object" or str(df[c].dtype).startswith("str"):
            df[c] = df[c].astype("category").cat.codes.astype("int32")
    return df


def _te_map(df: pd.DataFrame, idx: np.ndarray, prior: float) -> pd.Series:
    g = df.iloc[idx].groupby(ENTITY)[config.TARGET].agg(["sum", "count"])
    return (g["sum"] + prior * SMOOTH) / (g["count"] + SMOOTH)


def evaluate(df: pd.DataFrame, folds, global_te: bool, features: list[str]) -> float:
    import lightgbm as lgb

    prior = float(df[config.TARGET].mean())
    gmap = _te_map(df, np.arange(len(df)), prior) if global_te else None

    aucs = []
    for tr, va in folds:
        mapping = gmap if global_te else _te_map(df, tr, prior)
        te = df[ENTITY].map(mapping).fillna(prior).to_numpy(dtype="float32")
        X = df[features].copy()
        X["card_te"] = te
        y = df[config.TARGET].to_numpy()

        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=63,
            colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
            random_state=config.SEED, verbose=-1, n_jobs=-1,
        )
        model.fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[va], model.predict_proba(X.iloc[va])[:, 1]))
    return float(np.mean(aucs))


def run() -> pd.DataFrame:
    df = prepare()
    features = [c for c in df.columns if c not in DROP]
    print(f"rows={len(df):,}  features={len(features)}  "
          f"fraud={df[config.TARGET].mean():.3%}  entity={ENTITY} "
          f"({df[ENTITY].nunique():,} unique)")

    shuffled = split.random_kfold(df, n_folds=N_FOLDS)
    chrono = split.expanding_window_folds(df, n_folds=N_FOLDS)
    # The competition's test set starts 30 days after train ends. Contiguous
    # folds quietly validate on the day after the last training row, which is an
    # easier task than the one being scored.
    gapped = split.expanding_window_folds(df, n_folds=N_FOLDS, gap=30 * config.DAY)

    rows = []
    for sname, folds in (("shuffled K-fold", shuffled), ("chronological", chrono),
                         ("chronological + 30d gap", gapped)):
        for tname, gte in (("global", True), ("fold-local", False)):
            auc = evaluate(df, folds, gte, features)
            rows.append({
                "split": sname, "target encoding": tname, "AUC": round(auc, 4),
                "trains on future": split.max_train_time_beyond_val(df, *folds[0][:2]) > 0,
                "card overlap (fold 1)": round(
                    split.entity_overlap(df, folds[0][0], folds[0][1], ENTITY), 3),
            })
            print(f"  {sname:16s} + {tname:10s} TE -> AUC {auc:.4f}")

    out = pd.DataFrame(rows)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.REPORTS / "leakage_real.csv", index=False)

    honest = out.query("split=='chronological' and `target encoding`=='fold-local'")["AUC"].iloc[0]
    feat_leak = out.query("split=='chronological' and `target encoding`=='global'")["AUC"].iloc[0] - honest
    split_leak = out.query("split=='shuffled K-fold' and `target encoding`=='fold-local'")["AUC"].iloc[0] - honest
    print(f"\nhonest baseline           {honest:.4f}")
    print(f"feature leak (global TE)  +{feat_leak:.4f}")
    print(f"split leak (shuffled)     +{split_leak:.4f}")
    if split_leak > 0:
        print(f"ratio feature:split       {feat_leak / split_leak:.2f}x")
    gapped_honest = out.query(
        "split=='chronological + 30d gap' and `target encoding`=='fold-local'")["AUC"]
    if len(gapped_honest):
        print(f"with the real 30-day gap  {gapped_honest.iloc[0]:.4f} "
              f"({gapped_honest.iloc[0] - honest:+.4f} vs contiguous folds)")
    print(f"-> {config.REPORTS / 'leakage_real.csv'}")
    return out


if __name__ == "__main__":
    run()

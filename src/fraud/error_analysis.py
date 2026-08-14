"""Where the model fails, and whether the failures share a shape.

Predictions come from the honest setup -- chronological folds, 30-day embargo,
fold-local aggregates -- so these are the errors the model would actually make,
not the flattering ones a shuffled split produces.

Framed around an alert budget rather than a 0.5 threshold. Nobody reviews every
transaction scoring over 0.5; a fraud team works a queue of fixed size, so
"recall at the top 1%" is the number that decides whether the model is useful.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config, split, train

BUDGETS = (0.001, 0.005, 0.01, 0.05)


def oof_predictions() -> pd.DataFrame:
    """Out-of-fold scores under the honest split."""
    import lightgbm as lgb

    df = train.prepare()
    folds = split.expanding_window_folds(df, n_folds=train.N_FOLDS, gap=train.GAP)
    y = df[config.TARGET].to_numpy()
    pred = np.full(len(df), np.nan)

    for tr, va in folds:
        d, cols = train._build(df, tr, train.FINAL_GROUPS)
        m = lgb.LGBMClassifier(
            n_estimators=train.N_ESTIMATORS, learning_rate=0.05, num_leaves=63,
            colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
            min_child_samples=50, random_state=config.SEED, verbose=-1, n_jobs=-1,
        )
        m.fit(d.iloc[tr][cols], y[tr])
        pred[va] = m.predict_proba(d.iloc[va][cols])[:, 1]

    scored = df.loc[~np.isnan(pred)].copy()
    scored["pred"] = pred[~np.isnan(pred)]
    # Only validated rows have a score; the seed-history block never does.
    print(f"scored {len(scored):,} of {len(df):,} rows "
          f"({scored[config.TARGET].mean():.3%} fraud)")
    return scored


def recall_at_budget(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-p)
    rows = []
    for b in BUDGETS:
        k = max(1, int(len(p) * b))
        caught = y[order[:k]].sum()
        rows.append({
            "alert budget": f"{b:.1%}",
            "n reviewed": k,
            "fraud caught": int(caught),
            "recall": round(float(caught / y.sum()), 4),
            "precision": round(float(caught / k), 4),
        })
    return pd.DataFrame(rows)


def _seg(d: pd.DataFrame, name: str, key) -> list[dict]:
    rows = []
    for val, g in d.groupby(key, observed=True):
        if len(g) < 500 or g[config.TARGET].nunique() < 2:
            continue
        k = max(1, int(len(g) * 0.01))
        top = g.nlargest(k, "pred")
        rows.append({
            "segment": name,
            "value": str(val),
            "n": len(g),
            "fraud rate": round(float(g[config.TARGET].mean()), 4),
            "AUC": round(float(roc_auc_score(g[config.TARGET], g["pred"])), 4),
            "recall@1%": round(float(top[config.TARGET].sum() / g[config.TARGET].sum()), 3),
        })
    return rows


def run() -> None:
    d = oof_predictions()
    y, p = d[config.TARGET].to_numpy(), d["pred"].to_numpy()
    config.REPORTS.mkdir(parents=True, exist_ok=True)

    print(f"\noverall AUC {roc_auc_score(y, p):.4f}   "
          f"PR-AUC {average_precision_score(y, p):.4f}   "
          f"(a random ranker would score {y.mean():.4f} PR-AUC)")

    budget = recall_at_budget(y, p)
    budget.to_csv(config.REPORTS / "ea_budget.csv", index=False)
    print("\nrecall at a fixed review budget:")
    print(budget.to_string(index=False))

    # --- where is it weakest -------------------------------------------
    d["amt_decile"] = pd.qcut(d["TransactionAmt"], 10, labels=False, duplicates="drop")
    d["val_week"] = ((d[config.TIME_COL] - d[config.TIME_COL].min()) // (7 * config.DAY))
    rows = []
    rows += _seg(d, "ProductCD", "ProductCD")
    rows += _seg(d, "has_identity", "has_identity")
    rows += _seg(d, "amount decile", "amt_decile")
    seg = pd.DataFrame(rows).sort_values("AUC")
    seg.to_csv(config.REPORTS / "ea_segments.csv", index=False)
    print("\nweakest segments (AUC ascending):")
    print(seg.head(10).to_string(index=False))

    # --- does it decay over the validation window? ---------------------
    weeks = []
    for w, g in d.groupby("val_week", observed=True):
        if len(g) < 2000 or g[config.TARGET].nunique() < 2:
            continue
        weeks.append({"week": int(w), "n": len(g),
                      "fraud rate": round(float(g[config.TARGET].mean()), 4),
                      "AUC": round(float(roc_auc_score(g[config.TARGET], g["pred"])), 4)})
    wk = pd.DataFrame(weeks)
    wk.to_csv(config.REPORTS / "ea_drift.csv", index=False)
    print("\nAUC by week of the validation window:")
    print(wk.to_string(index=False))

    # --- calibration ---------------------------------------------------
    d["bucket"] = pd.cut(d["pred"], [0, .01, .05, .1, .25, .5, .75, 1.0],
                         include_lowest=True)
    cal = d.groupby("bucket", observed=True).agg(
        n=("pred", "size"), mean_pred=("pred", "mean"),
        actual=(config.TARGET, "mean")).round(4).reset_index()
    cal["bucket"] = cal["bucket"].astype(str)
    cal.to_csv(config.REPORTS / "ea_calibration.csv", index=False)
    print("\ncalibration:")
    print(cal.to_string(index=False))

    # --- the misses ----------------------------------------------------
    fraud = d[d[config.TARGET] == 1]
    missed = fraud.nsmallest(max(1, int(len(fraud) * 0.25)), "pred")
    caught = fraud.nlargest(max(1, int(len(fraud) * 0.25)), "pred")
    cmp = pd.DataFrame({
        "metric": ["median amount", "share with identity", "median C1",
                   "median D1", "share ProductCD=W"],
        "hardest 25% of fraud": [
            round(float(missed["TransactionAmt"].median()), 2),
            round(float(missed["has_identity"].mean()), 3),
            round(float(missed["C1"].median()), 1),
            round(float(missed["D1"].median()), 1),
            round(float((missed["ProductCD"] == d["ProductCD"].mode()[0]).mean()), 3)],
        "easiest 25%": [
            round(float(caught["TransactionAmt"].median()), 2),
            round(float(caught["has_identity"].mean()), 3),
            round(float(caught["C1"].median()), 1),
            round(float(caught["D1"].median()), 1),
            round(float((caught["ProductCD"] == d["ProductCD"].mode()[0]).mean()), 3)],
    })
    cmp.to_csv(config.REPORTS / "ea_missed.csv", index=False)
    print("\nfraud the model ranks lowest vs highest:")
    print(cmp.to_string(index=False))
    print(f"\n-> {config.REPORTS}")


if __name__ == "__main__":
    run()

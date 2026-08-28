"""Score the real competition test set and write a Kaggle submission.

Why this exists
---------------
Every other number in this repo is cross-validation on the training period. That
makes the headline claim, that 0.8513 is the honest estimate and 0.9557 is
self-flattery, untested against anything external. The competition's private
leaderboard is scored on a period beginning 30 days after training ends, which
is precisely the situation the 30-day-embargo CV was built to imitate. So the
leaderboard is the one check that can tell me whether my "honest" number was
actually honest, and it is not a check I can fool.

Prediction, written down before submitting: the private score should land near
**0.8513**, not near 0.9557. If it comes in far below 0.85, even the embargoed
estimate was optimistic and NOTES.md entry 5 needs revising.

Two traps this file has to avoid
--------------------------------
1. **Categorical codes must be fitted jointly.** `train.prepare()` encodes
   non-numeric columns with `.astype("category").cat.codes`, which numbers
   categories by sorted order *within the frame it is given*. Encode train and
   test separately and `ProductCD == "W"` can be 4 in one and 3 in the other:
   the columns still line up, the dtypes still match, nothing raises, and the
   submission is quietly garbage. Train and test are concatenated before
   encoding for this reason alone.
2. **Frequency counts must come from train only.** `add_frequency` maps counts
   gathered on `tr_idx` across the whole frame, so passing the training indices
   fits on train and applies to test, rather than letting the test set inform
   its own features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data, features
from .train import DROP, FREQ_COLS, N_ESTIMATORS


def build(out_path=None) -> "pd.DataFrame":
    import lightgbm as lgb

    print("loading train + test ...")
    tr = data.load_raw("train")
    te = data.load_raw("test")
    n_tr, n_te = len(tr), len(te)
    test_ids = te[config.ID_COL].to_numpy()
    print(f"  train {n_tr:,}   test {n_te:,}")

    te[config.TARGET] = np.nan
    both = pd.concat([tr, te], ignore_index=True, sort=False)
    del tr, te

    # Joint encoding: see trap 1 in the module docstring.
    for c in both.columns:
        if not pd.api.types.is_numeric_dtype(both[c]):
            both[c] = both[c].astype("category").cat.codes.astype("int32")

    both = features.base(both)
    both["_uid"] = features.uid(both)
    tr_idx = np.arange(n_tr)
    both = features.add_frequency(both, tr_idx, FREQ_COLS)

    cols = [c for c in both.columns if c not in DROP]
    y = both[config.TARGET].to_numpy()[:n_tr]
    print(f"  features {len(cols)}")

    # Same configuration as the ablation winner: engineered base + frequency,
    # no uid aggregates (+0.0004) and no target encoding (-0.0312).
    m = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS, learning_rate=0.05, num_leaves=63,
        colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
        min_child_samples=50, random_state=config.SEED, verbose=-1, n_jobs=-1,
    )
    print("fitting on the full training period ...")
    m.fit(both.iloc[:n_tr][cols], y)

    print("scoring test ...")
    p = m.predict_proba(both.iloc[n_tr:][cols])[:, 1]

    sub = pd.DataFrame({config.ID_COL: test_ids, config.TARGET: p})
    out_path = out_path or config.REPORTS / "submission.csv"
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"-> {out_path}  ({len(sub):,} rows, mean p={p.mean():.4f})")
    return sub


if __name__ == "__main__":
    build()

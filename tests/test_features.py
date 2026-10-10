import numpy as np
import pandas as pd

from src.fraud import config, features


def test_target_encoding_never_sees_a_rows_own_label():
    # Every key is unique, so the only information a training row's encoding
    # could carry about its label is the label itself. Out of fold, it must
    # fall back to the prior for every training row.
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"k": np.arange(n), config.TARGET: rng.integers(0, 2, n)})
    tr = np.arange(300)
    te = features.add_target_encoding(df, tr, ["k"])["k_te"].to_numpy()
    assert np.ptp(te[tr]) < 1e-6
    assert np.allclose(te[300:], df[config.TARGET].iloc[tr].mean())


def test_target_encoding_val_rows_use_all_training_rows():
    df = pd.DataFrame({"k": [0] * 10 + [0], config.TARGET: [1] * 10 + [0]})
    out = features.add_target_encoding(df, np.arange(10), ["k"])
    prior, s = 1.0, features.SMOOTH
    assert np.isclose(out["k_te"].iloc[10], (10 + prior * s) / (10 + s))

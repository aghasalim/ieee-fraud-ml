"""The splitters are the load-bearing part of this project's methodology, so
they get tested on synthetic data where the right answer is known by construction.
"""
import numpy as np
import pandas as pd
import pytest

from src.fraud import split


@pytest.fixture
def df():
    """300 rows, time strictly increasing, 30 recurring entities.

    Rows are deliberately NOT pre-sorted by time, so a splitter that assumes
    sorted input will fail these tests.
    """
    rng = np.random.default_rng(0)
    n = 300
    d = pd.DataFrame({
        "TransactionDT": np.arange(n) * 100,
        "card": rng.integers(0, 30, n),
        "y": rng.integers(0, 2, n),
    })
    return d.sample(frac=1.0, random_state=1).reset_index(drop=True)


def test_holdout_is_strictly_later_than_train(df):
    tr, va = split.time_ordered_holdout(df, frac=0.2)
    assert df["TransactionDT"].to_numpy()[tr].max() < df["TransactionDT"].to_numpy()[va].min()
    assert len(tr) + len(va) == len(df)
    assert len(va) == pytest.approx(len(df) * 0.2, abs=1)


def test_holdout_partitions_without_overlap(df):
    tr, va = split.time_ordered_holdout(df, frac=0.2)
    assert set(tr).isdisjoint(set(va))


def test_expanding_folds_never_train_on_the_future(df):
    for tr, va in split.expanding_window_folds(df, n_folds=4):
        assert split.max_train_time_beyond_val(df, tr, va) < 0


def test_expanding_folds_grow(df):
    sizes = [len(tr) for tr, _ in split.expanding_window_folds(df, n_folds=4)]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes), "each fold should add history"


def test_expanding_folds_validate_disjoint_blocks(df):
    vals = [set(va) for _, va in split.expanding_window_folds(df, n_folds=4)]
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            assert vals[i].isdisjoint(vals[j])


def test_random_kfold_does_leak_the_future(df):
    """Guards the comparison itself: if this ever stopped leaking, the headline
    experiment in NOTES.md would be measuring nothing."""
    leaks = [
        split.max_train_time_beyond_val(df, tr, va) > 0
        for tr, va in split.random_kfold(df, n_folds=4)
    ]
    assert all(leaks)


def test_random_kfold_covers_every_row_exactly_once(df):
    vals = np.concatenate([va for _, va in split.random_kfold(df, n_folds=4)])
    assert sorted(vals) == list(range(len(df)))


def test_entity_overlap_extremes():
    d = pd.DataFrame({"TransactionDT": range(10), "card": [1] * 5 + [2] * 5})
    tr, va = np.arange(5), np.arange(5, 10)
    assert split.entity_overlap(d, tr, va, "card") == 0.0
    assert split.entity_overlap(d, np.arange(10), np.arange(10), "card") == 1.0


def test_entity_overlap_ignores_nan_entities():
    d = pd.DataFrame({"TransactionDT": range(4), "card": [1, np.nan, 1, np.nan]})
    assert split.entity_overlap(d, np.array([0]), np.array([1, 2, 3]), "card") == 1.0

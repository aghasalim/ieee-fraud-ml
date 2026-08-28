"""Locks in the claim the README is built on.

If a refactor ever makes the leaky configurations stop looking better than the
honest one, the headline experiment is silently measuring nothing, and a
passing test suite would be actively misleading. Hence these run the real code
path rather than mocking it, just at a smaller size.
"""
import pytest

from src.fraud import config, split
from src.fraud.experiments import validation_gap as vg


@pytest.fixture(scope="module")
def df():
    return vg.make_synthetic(n=20_000, n_cards=1_500)


def test_synthetic_base_rate_resembles_fraud(df):
    """Severe imbalance is load-bearing: it governs how much a target encoding
    can memorise per card. A drift here invalidates the whole demonstration."""
    assert 0.02 < df["y"].mean() < 0.07


def test_synthetic_cards_recur(df):
    """Without recurring entities there is nothing for a target encoding to leak."""
    assert df["card"].value_counts().mean() > 5


def test_global_target_encoding_inflates_auc(df):
    """The central claim: a leaky *feature* flatters the score even when the
    split is chronologically correct."""
    chrono = split.expanding_window_folds(df, n_folds=3)
    leaky = vg.evaluate(df, chrono, global_te=True)
    honest = vg.evaluate(df, chrono, global_te=False)
    assert leaky > honest + 0.05


def test_shuffled_split_inflates_auc(df):
    """The better-known leak, isolated by holding the encoding fold-local."""
    folds_shuffled = split.random_kfold(df, n_folds=3)
    folds_chrono = split.expanding_window_folds(df, n_folds=3)
    assert vg.evaluate(df, folds_shuffled, global_te=False) > vg.evaluate(
        df, folds_chrono, global_te=False
    )


def test_honest_configuration_is_the_least_flattering(df):
    chrono = split.expanding_window_folds(df, n_folds=3)
    shuffled = split.random_kfold(df, n_folds=3)
    honest = vg.evaluate(df, chrono, global_te=False)
    others = [
        vg.evaluate(df, chrono, global_te=True),
        vg.evaluate(df, shuffled, global_te=True),
        vg.evaluate(df, shuffled, global_te=False),
    ]
    assert all(o > honest for o in others)

"""Validation splitting.

This module exists because the split *is* the experiment. Every score in this
repo is only as trustworthy as the partition that produced it, so the splitters
live in one small, tested file rather than being written inline in a notebook
where nobody can check them.

Two different mistakes get conflated under the word "leakage", and this project
is careful to keep them apart:

1. **Temporal leakage**, training on rows that occur *after* the rows being
   validated. This is unambiguously wrong here. The competition's test set is
   the period immediately following train, so a model validated on shuffled data
   is being asked an easier question than the one it will be scored on.

2. **Entity overlap**, the same card/device appearing in both train and
   validation. This one is *not* automatically a bug, and calling it one is a
   common overcorrection. At real inference time you genuinely do know a card's
   past behaviour, so a model that uses card history is doing something
   legitimate. It becomes a bug only when the aggregate is computed over the
   whole dataset, because then the training rows have absorbed statistics from
   the validation period, future information wearing a per-entity disguise.

So: temporal ordering is enforced structurally, and entity aggregates are
computed fold-locally (see `features.py`) rather than banned outright.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def time_ordered_holdout(
    df: pd.DataFrame, time_col: str = config.TIME_COL, frac: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Final holdout: the last `frac` of the data in time order.

    Returned as positional indices into `df` as given.
    """
    frac = config.HOLDOUT_FRAC if frac is None else frac
    order = np.argsort(df[time_col].to_numpy(), kind="stable")
    cut = int(len(order) * (1 - frac))
    return order[:cut], order[cut:]


def expanding_window_folds(
    df: pd.DataFrame,
    time_col: str = config.TIME_COL,
    n_folds: int | None = None,
    gap: float = 0.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Chronological CV: fold *i* trains on everything before a cut and
    validates on the block immediately after it.

    Expanding rather than sliding window: fraud patterns drift, but there is no
    evidence that old data becomes actively harmful here, and discarding history
    would leave the early folds training on very little.

    `gap` is an embargo, in the units of `time_col`, between the end of training
    and the start of validation. It defaults to 0, but 0 is *not* the honest
    setting for this competition: the real test set begins 30 days after train
    ends, so contiguous folds validate a model on the day after its last
    training row, an easier task than the one being scored. Passing
    `gap=30*config.DAY` reproduces the real handicap.
    """
    n_folds = config.N_FOLDS if n_folds is None else n_folds
    t = df[time_col].to_numpy()
    order = np.argsort(t, kind="stable")
    n = len(order)
    # n_folds+1 blocks: the first is train-only seed history, then each
    # subsequent block is validated once.
    bounds = np.linspace(0, n, n_folds + 2).astype(int)
    folds = []
    for i in range(1, n_folds + 1):
        tr = order[: bounds[i]]
        va = order[bounds[i] : bounds[i + 1]]
        if gap > 0 and len(va):
            tr = tr[t[tr] <= t[va].min() - gap]
        if len(va) and len(tr):
            folds.append((tr, va))
    return folds


def random_kfold(
    df: pd.DataFrame, n_folds: int | None = None, seed: int = config.SEED
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffled K-fold, deliberately WRONG for this dataset.

    Kept in the codebase on purpose: the gap between this and
    `expanding_window_folds` is the measurement that turns "you should use a
    time split" from advice into evidence. See NOTES.md.
    """
    n_folds = config.N_FOLDS if n_folds is None else n_folds
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    blocks = np.array_split(idx, n_folds)
    return [
        (np.concatenate([b for j, b in enumerate(blocks) if j != i]), blocks[i])
        for i in range(n_folds)
    ]


def entity_overlap(
    df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, entity_col: str
) -> float:
    """Fraction of validation entities that also appear in train.

    Diagnostic, not a pass/fail gate, see the module docstring on why some
    overlap is legitimate.
    """
    col = df[entity_col].to_numpy()
    tr = pd.unique(col[train_idx])
    va = pd.unique(col[val_idx])
    va = va[~pd.isna(va)]
    if len(va) == 0:
        return 0.0
    return float(np.isin(va, tr).mean())


def max_train_time_beyond_val(
    df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
    time_col: str = config.TIME_COL,
) -> float:
    """How far the training set reaches *past* the start of validation.

    Zero or negative means no temporal leakage. Positive means the model is
    seeing the future, and the magnitude says how much.
    """
    t = df[time_col].to_numpy()
    return float(t[train_idx].max() - t[val_idx].min())

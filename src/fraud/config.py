"""Central configuration. Every experiment reads its knobs from here so that a
number in NOTES.md can always be traced back to the settings that produced it.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

COMPETITION = "ieee-fraud-detection"
TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

SEED = 42

# TransactionDT is an offset in seconds from an undisclosed reference date, not a
# real timestamp. The competition's own train/test split is chronological: test
# begins where train ends. Any validation that does not respect that ordering is
# measuring a different problem than the one being scored.
DAY = 60 * 60 * 24

# Number of chronological folds for time-series CV.
N_FOLDS = int(os.getenv("N_FOLDS", "5"))

# Fraction of the (time-ordered) training data held out as the final, untouched
# validation period. Chosen to approximate the train/test gap in the competition.
HOLDOUT_FRAC = float(os.getenv("HOLDOUT_FRAC", "0.2"))

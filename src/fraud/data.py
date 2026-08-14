"""Download and load the IEEE-CIS Fraud Detection data.

The two tables are joined left-on-transaction, never inner. Only about a quarter
of transactions carry an identity record, and *whether* identity is present is
itself predictive -- an inner join would silently drop three quarters of the
data and quietly change the problem.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from . import config


def _credentials_present() -> bool:
    """Accept any of the three shapes Kaggle credentials currently take.

    Kaggle moved from a `kaggle.json` username+key pair to a single `KGAT_`
    bearer token, so checking only for the legacy file rejects a perfectly valid
    modern setup. The older file still works, hence all three.
    """
    if os.getenv("KAGGLE_API_TOKEN") or os.getenv("KAGGLE_KEY"):
        return True
    kdir = Path.home() / ".kaggle"
    return any((kdir / name).exists() for name in ("access_token", "kaggle.json"))


def download() -> None:
    """Fetch the competition archive via the Kaggle CLI.

    Requires ~/.kaggle/kaggle.json and acceptance of the competition rules. Both
    are the user's to provide; this function only reports clearly when they are
    missing, because the raw Kaggle error ("403 Forbidden") does not distinguish
    "no token" from "rules not accepted".
    """
    config.RAW.mkdir(parents=True, exist_ok=True)
    if not _credentials_present():
        sys.exit(
            "No Kaggle credentials found.\n"
            "  kaggle.com -> Settings -> API -> Create New API Token, then either:\n"
            "    export KAGGLE_API_TOKEN=KGAT_...\n"
            "  or:\n"
            "    mkdir -p ~/.kaggle && echo 'KGAT_...' > ~/.kaggle/access_token"
            " && chmod 600 ~/.kaggle/access_token"
        )

    # The CLI ships inside the virtualenv, which is not on PATH when this module
    # is run as `python -m src.fraud.data`. Resolve it next to the interpreter
    # that is actually running rather than trusting the ambient PATH.
    kaggle_bin = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")

    archive = config.RAW / f"{config.COMPETITION}.zip"
    if not archive.exists():
        proc = subprocess.run(
            [kaggle_bin, "competitions", "download", "-c", config.COMPETITION,
             "-p", str(config.RAW)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip()
            if "403" in err or "Forbidden" in err:
                sys.exit(
                    "Kaggle returned 403. The token works but the competition "
                    "rules have not been accepted for this account.\n"
                    f"  Accept them at https://www.kaggle.com/c/{config.COMPETITION}/rules"
                )
            sys.exit(f"kaggle download failed:\n{err}")

    with zipfile.ZipFile(archive) as z:
        z.extractall(config.RAW)
    print(f"extracted -> {config.RAW}")


def reduce_mem(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Downcast numeric columns to the narrowest dtype that holds their range.

    train_transaction is 590k x 394; at float64 that is ~1.8 GB before a single
    feature is built. Downcasting keeps the whole pipeline comfortably in RAM.

    Floats go to float32, never float16: float16 has ~3 decimal digits of
    precision, and TransactionAmt carries cents that matter (the decimal part of
    the amount turns out to be a real signal).
    """
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        t = df[col].dtype
        # Test numeric-ness rather than matching dtype names. pandas 3.0 stores
        # strings as a `str` dtype instead of `object`, so a name-based check
        # silently lets ProductCD through and dies on astype("float32").
        if not pd.api.types.is_numeric_dtype(t) or pd.api.types.is_bool_dtype(t):
            continue
        if str(t).startswith("int"):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        else:
            df[col] = df[col].astype("float32")
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"  memory {start:.0f} MB -> {end:.0f} MB ({1 - end / start:.0%} smaller)")
    return df


def load_raw(split: str = "train", nrows: int | None = None) -> pd.DataFrame:
    """Load one split, joining identity onto transaction."""
    tx = pd.read_csv(config.RAW / f"{split}_transaction.csv", nrows=nrows)
    idf = pd.read_csv(config.RAW / f"{split}_identity.csv")
    # test_identity uses id-01 style column names while train_identity uses
    # id_01; unifying them here avoids a silent all-NaN block at predict time.
    idf.columns = [c.replace("-", "_") for c in idf.columns]
    df = tx.merge(idf, on=config.ID_COL, how="left")
    df["has_identity"] = df[config.ID_COL].isin(idf[config.ID_COL]).astype("int8")
    return reduce_mem(df, verbose=True)


if __name__ == "__main__":
    download()

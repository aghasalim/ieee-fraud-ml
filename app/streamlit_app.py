"""Interactive fraud predictor with a per-prediction SHAP explanation."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fraud import config  # noqa: E402

st.set_page_config(page_title="Fraud predictor", page_icon="🔍", layout="wide")


@st.cache_resource
def load_model():
    p = config.ARTIFACTS / "model.pkl"
    return joblib.load(p) if p.exists() else None


bundle = load_model()
st.title("IEEE-CIS fraud predictor")

if bundle is None:
    st.error("No model found. Run `make data && make train-final` first.")
    st.stop()

st.caption(
    f"LightGBM over {len(bundle['columns'])} features, trained on "
    f"{bundle['n_train']:,} transactions. Validation AUC **{bundle['val_auc']:.4f}** "
    "under chronological folds with a 30-day embargo — the honest number, not the "
    "0.9557 a shuffled split would have reported."
)

st.warning(
    "**This form supplies 16 of 443 features.** Everything else is filled with "
    "population medians, so treat the output as a demonstration of the model and "
    "its explanation, not a production score. The V-columns are anonymised "
    "engineered features from Vesta that carry much of the real signal and cannot "
    "meaningfully be typed into a form."
)

defaults = bundle["defaults"]
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Transaction")
    amt = st.number_input("amount", 0.0, 50_000.0, 120.0, step=10.0)
    product = st.selectbox("product code", ["W", "C", "R", "H", "S"], index=0)
    hour = st.slider("hour of day", 0, 23, 14)
    dow = st.slider("day of week", 0, 6, 2)
with c2:
    st.subheader("Card / address")
    card1 = st.number_input("card1 id", 1000, 20_000, 7919)
    card4 = st.selectbox("network", ["visa", "mastercard", "amex", "discover"])
    card6 = st.selectbox("type", ["debit", "credit"])
    addr1 = st.number_input("addr1", 100, 550, 299)
with c3:
    st.subheader("History / identity")
    c1_count = st.number_input("C1 (card count)", 0, 5000, 1)
    c13 = st.number_input("C13", 0, 5000, 1)
    d1 = st.number_input("D1 (days since first txn)", 0, 700, 0)
    has_id = st.checkbox("identity record present", value=False)

# Product code mapping matches the category ordering used at training time
# (pandas sorts categories alphabetically before .cat.codes).
PRODUCT = {c: i for i, c in enumerate(sorted(["C", "H", "R", "S", "W"]))}
CARD4 = {c: i for i, c in enumerate(sorted(["american express", "discover",
                                            "mastercard", "visa"]))}
CARD6 = {c: i for i, c in enumerate(sorted(["credit", "debit"]))}
NET = {"visa": "visa", "mastercard": "mastercard", "amex": "american express",
       "discover": "discover"}

row = pd.DataFrame([defaults])[bundle["columns"]]
row.loc[0, "TransactionAmt"] = amt
row.loc[0, "amt_log"] = float(np.log1p(amt))
row.loc[0, "amt_cents"] = round((amt - np.floor(amt)) * 100)
row.loc[0, "amt_is_round"] = int(round((amt - np.floor(amt)) * 100) == 0)
row.loc[0, "ProductCD"] = PRODUCT[product]
row.loc[0, "hour"] = hour
row.loc[0, "dayofweek"] = dow
row.loc[0, "card1"] = card1
row.loc[0, "card4"] = CARD4[NET[card4]]
row.loc[0, "card6"] = CARD6[card6]
row.loc[0, "addr1"] = addr1
row.loc[0, "C1"] = c1_count
row.loc[0, "C13"] = c13
row.loc[0, "D1"] = d1
row.loc[0, "has_identity"] = int(has_id)
row = row.astype("float32")

model = bundle["model"]
p = float(model.predict_proba(row)[0, 1])

st.divider()
m1, m2 = st.columns([1, 2])
m1.metric("fraud probability", f"{p:.2%}")
base_rate = 0.03499
m1.caption(f"base rate {base_rate:.2%} — this is {p/base_rate:.1f}× baseline")
if p > 0.5:
    m1.error("would be flagged")
elif p > base_rate * 3:
    m1.warning("elevated")
else:
    m1.success("looks ordinary")

with m2:
    st.subheader("Why — SHAP contributions")
    try:
        import shap

        expl = shap.TreeExplainer(model)
        vals = expl.shap_values(row)
        v = vals[1][0] if isinstance(vals, list) else np.asarray(vals)[0]
        s = pd.Series(np.ravel(v), index=bundle["columns"])
        top = s.reindex(s.abs().sort_values(ascending=False).index)[:12]
        chart = pd.DataFrame({"feature": top.index, "contribution": top.to_numpy()})
        st.bar_chart(chart.set_index("feature"), horizontal=True)
        st.caption(
            "Positive pushes toward fraud. Features you did not set still "
            "contribute, from their median value — which is why the anonymised "
            "C, D and identity columns usually outrank the ones you can type. "
            "They hold signal the form cannot expose, and that is the honest "
            "limitation of a 16-field demo rather than a flaw in the model."
        )
    except Exception as e:  # shap is optional; the prediction still works
        st.info(f"SHAP unavailable ({type(e).__name__}). Prediction above is unaffected.")

st.divider()
st.markdown(
    "The point of this project is **[NOTES.md](https://github.com/aghasalim/"
    "ieee-fraud-ml/blob/main/NOTES.md)** — the decision trail, including the "
    "feature that backfired and a hypothesis I measured and withdrew."
)

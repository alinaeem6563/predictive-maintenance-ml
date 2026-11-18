"""Streamlit educational prototype. Run: streamlit run app/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from src import config as C

st.set_page_config(page_title="Predictive Maintenance Prototype", page_icon="🛠️")
st.title("🛠️ Predictive Maintenance – Failure Classifier")
st.warning("Educational prototype trained on the SYNTHETIC UCI AI4I 2020 dataset. "
           "Not a validated industrial maintenance or safety system.")

if not C.MODEL_PATH.exists():
    st.error("Model file not found. Run `python -m src.download_data` and `python -m src.train` first.")
    st.stop()


@st.cache_resource
def load_bundle():
    return joblib.load(C.MODEL_PATH)   # trusted, self-created file only


bundle = load_bundle()
data = pd.read_csv(C.MODEL_DATA)

st.subheader("Enter operating conditions")
mtype = st.selectbox("Type", ["L", "M", "H"])
vals = {}
for col in C.NUMERIC:
    lo, hi, med = float(data[col].min()), float(data[col].max()), float(data[col].median())
    vals[col] = st.number_input(col, min_value=lo, max_value=hi, value=med)

if st.button("Predict"):
    row = pd.DataFrame([{"Type": mtype, **vals}])[bundle["features"]]
    score = float(bundle["pipeline"].predict_proba(row)[0, 1])
    thr = bundle["threshold"]
    st.metric("Model score (class 1)", f"{score:.3f}")
    st.write(f"Decision threshold (selected on validation set): **{thr:.3f}**")
    if score >= thr:
        st.error("Predicted class: 1 – FAILURE likely (inspection suggested)")
    else:
        st.success("Predicted class: 0 – no failure predicted")
    st.caption("The score is a model output, not a calibrated real-world probability.")

with st.expander("About this model"):
    st.write(f"Model: **{bundle['model_name']}**. Predicts the dataset's row-level "
             "'Machine failure' label (0/1). It does not estimate remaining useful life "
             "or time to failure. Input ranges come from the training data and are not "
             "physical safety limits.")

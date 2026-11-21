"""Streamlit user interface for the Adult Income model (Question 3, step 8).

Start the API first, then (from the project root):
    streamlit run Q3/app/streamlit_app.py
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Adult Income Predictor", page_icon="💰", layout="wide")
st.title("💰 Adult Income Predictor")
st.caption("Predicts whether a person's yearly income exceeds $50,000 · model served through a FastAPI backend")


def api_get(path):
    r = requests.get(f"{API_URL}{path}", timeout=15)
    r.raise_for_status()
    return r.json()


try:
    health = api_get("/health")
    metrics = api_get("/metrics")
    importance = api_get("/feature-importance?top_k=5")
    schema = api_get("/schema")
except Exception as exc:
    st.error(f"Cannot reach the API at {API_URL}. Start it with:  uvicorn main:app --port 8000 --app-dir Q3/api\n\n{exc}")
    st.stop()

# ---------------------------------------------------------------- sidebar: model info
with st.sidebar:
    st.header("Model")
    st.write(f"**{metrics['model_name']}**")
    m = metrics["test_metrics"]
    st.subheader("Test-set performance")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c1.metric("F1", f"{m['f1']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    c1.metric("Recall", f"{m['recall']:.3f}")
    st.subheader("Top 5 important features")
    imp = pd.DataFrame(importance["top_features"])
    st.dataframe(imp.rename(columns={"feature": "Feature", "importance": "Importance"}), hide_index=True, width="stretch")
    st.caption(f"Importance = {importance['method']}")

# ---------------------------------------------------------------- main: input form
st.subheader("Enter a person's details")
num, cat = schema["numeric"], schema["categorical"]


def select(label, key):
    opts = cat[key]["options"]
    return st.selectbox(label, opts, index=opts.index(cat[key]["default"]))


col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", 17, 100, int(num["age"]["median"]))
    education_num = st.slider("Education (years, 1-16)", 1, 16, int(num["education_num"]["median"]))
    hours = st.slider("Hours per week", 1, 99, int(num["hours_per_week"]["median"]))
    sex = select("Sex", "sex")
with col2:
    workclass = select("Work class", "workclass")
    occupation = select("Occupation", "occupation")
    marital = select("Marital status", "marital_status")
    relationship = select("Relationship", "relationship")
with col3:
    race = select("Race", "race")
    country = select("Native country", "native_country")
    gain = st.number_input("Capital gain ($)", 0, 100000, 0, step=500)
    loss = st.number_input("Capital loss ($)", 0, 5000, 0, step=100)

payload = {"age": int(age), "workclass": workclass, "education_num": int(education_num), "marital_status": marital,
           "occupation": occupation, "relationship": relationship, "race": race, "sex": sex,
           "capital_gain": int(gain), "capital_loss": int(loss), "hours_per_week": int(hours), "native_country": country}

if st.button("Predict income", type="primary"):
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        out = r.json()
        p = out["probability_over_50k"]
        if out["prediction"] == 1:
            st.success(f"Predicted income: **more than $50K** (probability {p:.1%})")
        else:
            st.info(f"Predicted income: **$50K or less** (probability of >$50K: {p:.1%})")
        st.progress(min(max(p, 0.0), 1.0))
        with st.expander("Request sent to the API / raw response"):
            st.json({"request": payload, "response": out})
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

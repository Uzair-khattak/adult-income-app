"""Shared code for Question 3 (Adult Income): data loading, feature engineering,
preprocessing pipeline and metric helpers. It is imported by the notebooks, by the
FastAPI service and by the Streamlit app (the saved model references `add_features`)."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "income"
RANDOM_STATE = 42

# raw input columns the model expects (what the API / UI must provide)
RAW_NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
RAW_CATEGORICAL = ["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
INPUT_COLUMNS = RAW_NUMERIC + RAW_CATEGORICAL

# columns after feature engineering
NUM_COLS = RAW_NUMERIC + ["capital_net", "log_capital_gain", "has_capital_gain", "has_capital_loss", "is_married", "is_us"]
CAT_COLS = RAW_CATEGORICAL + ["age_group", "hours_category"]


# ------------------------------------------------------------------ loading
def _normalise(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace("-", "_").replace(".", "_").replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"class": "income", "educational_num": "education_num", "gender": "sex"})
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) in ("category", "string"):
            df[c] = df[c].astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)
            df[c] = df[c].replace({"?": np.nan})
    return df


def load_adult(data_dir):
    """Load the Adult Income dataset from data/q3_adult/*.csv. If no csv is there, try downloading
    it from OpenML (needs internet). Returns the cleaned raw DataFrame including the binary target."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.csv"))
    source = None
    if files:
        df = pd.read_csv(files[0], skipinitialspace=True)
        source = files[0].name
    else:
        from sklearn.datasets import fetch_openml
        print("No csv found in", data_dir, "-> downloading 'adult' from OpenML ...")
        df = fetch_openml("adult", version=2, as_frame=True, parser="auto").frame
        source = "OpenML adult v2"
    df = _normalise(df)
    df[TARGET] = df[TARGET].astype(str).str.contains(">50K").astype(int)
    df = df.drop(columns=[c for c in ["fnlwgt", "education"] if c in df.columns])   # fnlwgt = census weight, education duplicates education_num
    df.attrs["source"] = source
    return df


def split_data(df, test_size=0.2):
    X = df[INPUT_COLUMNS].copy()
    y = df[TARGET].astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


# ------------------------------------------------------------------ feature engineering
def add_features(df):
    """Derived features (works on a DataFrame that has the raw INPUT_COLUMNS)."""
    d = df.copy()
    d["capital_net"] = d["capital_gain"] - d["capital_loss"]
    d["log_capital_gain"] = np.log1p(d["capital_gain"].clip(lower=0))
    d["has_capital_gain"] = (d["capital_gain"] > 0).astype(int)
    d["has_capital_loss"] = (d["capital_loss"] > 0).astype(int)
    d["is_married"] = d["marital_status"].astype(str).str.startswith("Married").astype(int)
    d["is_us"] = (d["native_country"] == "United-States").astype(int)
    d["age_group"] = pd.cut(d["age"], [-1, 24, 34, 44, 54, 64, 200],
                            labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(object)
    d["hours_category"] = pd.cut(d["hours_per_week"], [-1, 34, 40, 50, 200],
                                 labels=["part_time", "full_time", "overtime", "heavy_overtime"]).astype(object)
    return d


def build_preprocessor():
    """feature engineering -> (impute + scale numeric) and (impute + one-hot categorical)."""
    columns = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUM_COLS),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT_COLS),
    ])
    return Pipeline([("features", FunctionTransformer(add_features)), ("columns", columns)])


# ------------------------------------------------------------------ evaluation helpers
def compute_metrics(model, X, y):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    return {"accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred)),
            "recall": float(recall_score(y, pred)), "f1": float(f1_score(y, pred)), "roc_auc": float(roc_auc_score(y, proba))}


def describe_complexity(pipe):
    """Rough, human-readable complexity measures of the final estimator in a fitted Pipeline."""
    est = pipe.steps[-1][1]
    info = {"final_estimator": type(est).__name__}
    inner = est
    if isinstance(est, Pipeline):
        info["n_steps_in_automl_pipeline"] = len(est.steps)
        inner = est.steps[-1][1]
        info["final_estimator"] = " -> ".join(type(s).__name__ for _, s in est.steps)
    if hasattr(inner, "coef_"):
        info["size"] = f"{np.asarray(inner.coef_).size} coefficients"
    elif hasattr(inner, "estimators_") and hasattr(np.asarray(inner.estimators_, dtype=object).ravel()[0], "tree_"):
        n_nodes = sum(t.tree_.node_count for t in np.asarray(inner.estimators_, dtype=object).ravel())
        info["size"] = f"{len(np.asarray(inner.estimators_, dtype=object).ravel())} trees, {n_nodes} nodes"
    elif hasattr(inner, "tree_"):
        info["size"] = f"1 tree, {inner.tree_.node_count} nodes"
    else:
        info["size"] = "n/a"
    return info


def time_inference(model, X, n_rows=5000):
    Xs = X.sample(min(n_rows, len(X)), random_state=0)
    t = time.perf_counter()
    model.predict(Xs)
    return (time.perf_counter() - t) / len(Xs) * 1000.0   # ms per row


def build_schema(X):
    """Allowed values for the UI / API (taken from the training data)."""
    schema = {"numeric": {}, "categorical": {}}
    for c in RAW_NUMERIC:
        schema["numeric"][c] = {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())}
    for c in RAW_CATEGORICAL:
        vals = X[c].dropna().astype(str)
        schema["categorical"][c] = {"options": sorted(vals.unique().tolist()), "default": vals.mode().iloc[0]}
    return schema

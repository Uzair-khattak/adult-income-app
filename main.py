"""FastAPI service for the Adult Income model (Question 3, step 7).

Run locally (from the project root, with the virtual environment active):
    uvicorn main:app --reload --port 8000 --app-dir Q3/api
Interactive docs:  http://127.0.0.1:8000/docs
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# make `q3_common` importable (the saved model needs it) both locally and in the deployment bundle
for p in [HERE, HERE.parent, HERE.parent.parent / "src", HERE.parent.parent]:
    sys.path.insert(0, str(p))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import q3_common  # noqa: F401  (needed so joblib can unpickle add_features)


def _find_model_dir():
    if os.environ.get("MODEL_DIR"):
        return Path(os.environ["MODEL_DIR"])
    for c in [HERE / "models", HERE.parent / "models", HERE.parent.parent / "models"]:
        if (c / "best_model.joblib").exists():
            return c
    raise FileNotFoundError("best_model.joblib not found. Run Q3/04_Comparison_and_Save_Model.ipynb first.")


MODEL_DIR = _find_model_dir()
MODEL = joblib.load(MODEL_DIR / "best_model.joblib")
META = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))

app = FastAPI(title="Adult Income Prediction API",
              description="Predicts whether a person earns more than $50K per year.", version="1.0")


class Person(BaseModel):
    age: int = Field(..., ge=17, le=100, examples=[39])
    workclass: str = Field(..., examples=["Private"])
    education_num: int = Field(..., ge=1, le=16, examples=[13])
    marital_status: str = Field(..., examples=["Never-married"])
    occupation: str = Field(..., examples=["Prof-specialty"])
    relationship: str = Field(..., examples=["Not-in-family"])
    race: str = Field(..., examples=["White"])
    sex: str = Field(..., examples=["Male"])
    capital_gain: int = Field(0, ge=0)
    capital_loss: int = Field(0, ge=0)
    hours_per_week: int = Field(..., ge=1, le=99, examples=[40])
    native_country: str = Field(..., examples=["United-States"])


@app.get("/")
def root():
    return {"message": "Adult Income API is running. See /docs for the interactive documentation."}


@app.get("/health")
def health():
    return {"status": "ok", "model": META["model_name"]}


@app.get("/metrics")
def metrics():
    """Test-set performance of the deployed model."""
    return {"model_name": META["model_name"], "test_metrics": META["test_metrics"]}


@app.get("/feature-importance")
def feature_importance(top_k: int = 5):
    return {"method": META["importance_method"], "top_features": META["feature_importance"][:top_k]}


@app.get("/schema")
def schema():
    return META["schema"]


@app.post("/predict")
def predict(person: Person):
    row = pd.DataFrame([person.model_dump()])[q3_common.INPUT_COLUMNS]
    try:
        proba = float(MODEL.predict_proba(row)[0, 1])
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
    pred = int(proba >= 0.5)
    return {"prediction": pred, "label": ">50K" if pred else "<=50K", "probability_over_50k": round(proba, 4)}

from fastapi import FastAPI
from pydantic import RootModel
import pickle
import pandas as pd
import numpy as np
import os

# -------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model_xgb_optimized.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Feature names from training
EXPECTED_FEATURES = model.named_steps["preprocess"].feature_names_in_

# Identify which features are categorical (object dtype) from training data
# We detect them from the fitted preprocess ColumnTransformer
preprocess = model.named_steps["preprocess"]
categorical_features = []
numeric_features = []

for name, transformer, columns in preprocess.transformers_:
    if name.startswith("cat"):
        categorical_features.extend(columns)
    elif name.startswith("num"):
        numeric_features.extend(columns)


# -------------------------------------------------------------------
# Request schema (Pydantic RootModel)
# -------------------------------------------------------------------
class HouseData(RootModel[dict]):
    """Incoming JSON payload: entire feature dictionary"""
    pass


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="House Price Prediction API",
    description="Predicts house sale price using an optimized XGBoost model",
    version="1.0"
)


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.post("/predict")
def predict(data: HouseData):

    # Extract dictionary from Pydantic v2 RootModel
    payload = data.root

    # Convert to DataFrame
    df = pd.DataFrame([payload])

    # Add missing columns with appropriate defaults
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            # Categorical → "missing"
            if col in categorical_features:
                df[col] = "missing"
            else:  # Numerical → NaN
                df[col] = np.nan

    # Replace None with proper values
    for col in categorical_features:
        if df[col].isna().any():
            df[col] = df[col].fillna("missing")

    for col in numeric_features:
        if df[col].isna().any():
            df[col] = df[col].astype(float).fillna(np.nan)

    # Ensure column order
    df = df[EXPECTED_FEATURES]

    # Predict log price
    pred_log = model.predict(df)[0]

    # Convert log1p back to actual price
    predicted_price = np.expm1(pred_log)

    return {
        "predicted_price": float(predicted_price),
        "predicted_price_rounded": round(float(predicted_price), 2)
    }

from pathlib import Path
import json
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
META_PATH = MODEL_DIR / "feature_order.json"

app = FastAPI(title="Taxi Demand Forecast API", version="1.0.0")


class PredictRequest(BaseModel):
    hour: int = Field(ge=0, le=23)
    dayofweek: int = Field(ge=0, le=6, description="0=Mon")
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)


def load_artifacts():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise RuntimeError("Model artifacts not found. Run: python src/train_baseline.py")

    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest):
    model, meta = load_artifacts()

    X = pd.DataFrame([
        {
            "hour": payload.hour,
            "dayofweek": payload.dayofweek,
            "month": payload.month,
            "day": payload.day,
        }
    ])

    y_pred = float(model.predict(X)[0])
    return {
        "model": meta.get("model", "unknown"),
        "features": X.to_dict(orient="records")[0],
        "predicted_trip_count": round(y_pred, 4),
    }

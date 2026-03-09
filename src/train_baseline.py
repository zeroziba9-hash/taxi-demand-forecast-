from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "train.csv"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

FEATURES = ["hour", "dayofweek", "month", "day"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime", "trip_count"])

    df["hour"] = df["pickup_datetime"].dt.hour
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    return df.sort_values("pickup_datetime")


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = add_time_features(df)

    X = df[FEATURES]
    y = df["trip_count"]

    # Time-based split (no shuffle)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    }

    rows = []
    pred_for_plot = None
    best_model_name = None
    best_model_obj = None
    best_rmse = float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae, rmse = evaluate(y_test, pred)
        rows.append({"model": name, "MAE": mae, "RMSE": rmse, "rows": len(df)})

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model_obj = model

        if name == "XGBoost":
            pred_for_plot = pred

        print(f"[{name}] MAE={mae:.4f}, RMSE={rmse:.4f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(rows).sort_values("RMSE")
    result_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(result_df["model"], result_df["RMSE"])
    plt.title("Model RMSE Comparison")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "rmse_comparison.png", dpi=150)
    plt.close()

    if pred_for_plot is not None:
        vis = pd.DataFrame({
            "actual": y_test.values[:200],
            "pred": pred_for_plot[:200],
        })
        plt.figure(figsize=(9, 4))
        plt.plot(vis["actual"].values, label="actual")
        plt.plot(vis["pred"].values, label="pred")
        plt.title("XGBoost Prediction vs Actual (first 200 test rows)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "prediction_preview.png", dpi=150)
        plt.close()

    if best_model_obj is not None:
        joblib.dump(best_model_obj, MODEL_DIR / "best_model.pkl")
        with open(MODEL_DIR / "feature_order.json", "w", encoding="utf-8") as f:
            json.dump({"features": FEATURES, "model": best_model_name}, f, ensure_ascii=False, indent=2)

    print("=== Training Done ===")
    print(f"Best model: {best_model_name} (RMSE={best_rmse:.4f})")
    print(f"Saved: {REPORT_DIR / 'model_comparison.csv'}")
    print(f"Saved: {REPORT_DIR / 'rmse_comparison.png'}")
    print(f"Saved: {REPORT_DIR / 'prediction_preview.png'}")
    print(f"Saved: {MODEL_DIR / 'best_model.pkl'}")


if __name__ == "__main__":
    main()

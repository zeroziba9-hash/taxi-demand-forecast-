from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "train.csv"
REPORT_DIR = BASE_DIR / "reports"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime", "trip_count"]).sort_values("pickup_datetime")
    df["hour"] = df["pickup_datetime"].dt.hour
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    return df


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = add_time_features(df)

    X = df[["hour", "dayofweek", "month", "day"]]
    y = df["trip_count"]

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

    tscv = TimeSeriesSplit(n_splits=5)
    rows = []

    for model_name, model in models.items():
        fold = 1
        fold_mae = []
        fold_rmse = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            m = mean_absolute_error(y_test, pred)
            r = rmse(y_test, pred)
            fold_mae.append(m)
            fold_rmse.append(r)

            rows.append({
                "model": model_name,
                "fold": fold,
                "MAE": m,
                "RMSE": r,
            })
            fold += 1

        rows.append({
            "model": model_name,
            "fold": "avg",
            "MAE": float(np.mean(fold_mae)),
            "RMSE": float(np.mean(fold_rmse)),
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "cv_metrics.csv", index=False)
    print(f"Saved: {REPORT_DIR / 'cv_metrics.csv'}")


if __name__ == "__main__":
    main()

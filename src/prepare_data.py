from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw.csv"
OUT_PATH = BASE_DIR / "data" / "train.csv"


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    # Expected raw schema (example: NYC taxi style)
    # pickup_datetime column candidates
    dt_candidates = [
        "pickup_datetime",
        "tpep_pickup_datetime",
        "lpep_pickup_datetime",
    ]

    dt_col = next((c for c in dt_candidates if c in df.columns), None)
    if dt_col is None:
        raise ValueError(f"No datetime column found. tried={dt_candidates}")

    df["pickup_datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=["pickup_datetime"])

    # If trip_count doesn't exist, aggregate to hourly demand
    if "trip_count" not in df.columns:
        out = (
            df.assign(hour=df["pickup_datetime"].dt.floor("h"))
            .groupby("hour", as_index=False)
            .size()
            .rename(columns={"hour": "pickup_datetime", "size": "trip_count"})
        )
    else:
        out = df[["pickup_datetime", "trip_count"]].copy()

    out = out.sort_values("pickup_datetime")
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} ({len(out)} rows)")


if __name__ == "__main__":
    main()

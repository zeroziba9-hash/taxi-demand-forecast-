from pathlib import Path
import json
import streamlit as st
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Taxi Demand Forecast Dashboard", layout="wide")
st.title("Taxi Demand Forecast Dashboard")

st.markdown("Baseline model comparison and prediction preview")

comp_path = REPORT_DIR / "model_comparison.csv"
rmse_img = REPORT_DIR / "rmse_comparison.png"
pred_img = REPORT_DIR / "prediction_preview.png"
model_path = MODEL_DIR / "best_model.pkl"
feature_path = MODEL_DIR / "feature_order.json"

if comp_path.exists():
    st.subheader("Model Comparison")
    df = pd.read_csv(comp_path)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("model_comparison.csv not found. Run: python src/train_baseline.py")

col1, col2 = st.columns(2)
with col1:
    st.subheader("RMSE Comparison")
    if rmse_img.exists():
        st.image(str(rmse_img), use_container_width=True)
    else:
        st.info("rmse_comparison.png not found")

with col2:
    st.subheader("Prediction Preview")
    if pred_img.exists():
        st.image(str(pred_img), use_container_width=True)
    else:
        st.info("prediction_preview.png not found")

st.markdown("---")
st.subheader("Cross Validation (TimeSeriesSplit)")
cv_path = REPORT_DIR / "cv_metrics.csv"
if cv_path.exists():
    cv_df = pd.read_csv(cv_path)
    st.dataframe(cv_df, use_container_width=True)
else:
    st.info("cv_metrics.csv not found. Run: python src/train_with_cv.py")

st.markdown("---")
st.subheader("Quick Inference")

if model_path.exists() and feature_path.exists():
    model = joblib.load(model_path)
    with open(feature_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        hour = st.number_input("hour", min_value=0, max_value=23, value=9)
    with c2:
        dayofweek = st.number_input("dayofweek (0=Mon)", min_value=0, max_value=6, value=0)
    with c3:
        month = st.number_input("month", min_value=1, max_value=12, value=1)
    with c4:
        day = st.number_input("day", min_value=1, max_value=31, value=1)

    if st.button("Predict trip_count"):
        X = pd.DataFrame([
            {
                "hour": hour,
                "dayofweek": dayofweek,
                "month": month,
                "day": day,
            }
        ])
        pred = float(model.predict(X)[0])
        st.success(f"Predicted trip_count: {pred:.2f}")
        st.caption(f"Model: {meta.get('model', 'unknown')}")
else:
    st.info("Model not found. Run: python src/train_baseline.py")

st.markdown("---")
st.caption("Run command: streamlit run app.py")

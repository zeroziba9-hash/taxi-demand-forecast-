# taxi-demand-forecast

NYC Taxi 수요 예측 프로젝트입니다.  
A practical **time-series demand forecasting** project with ML baselines + dashboard.

---

## 1) What this project does (프로젝트 목적)

- 택시 수요(`trip_count`)를 시간 기준으로 예측합니다.
- 두 모델을 비교합니다:
  - **RandomForestRegressor**
  - **XGBoostRegressor**
- 결과를 파일(csv/png)로 저장하고, Streamlit 대시보드에서 확인합니다.

---

## 2) Model Comparison (비교 모델)

| Model | Type | Why used |
|---|---|---|
| RandomForest | Tree Ensemble (Bagging) | baseline으로 안정적, 해석 쉬움 |
| XGBoost | Gradient Boosting | 성능이 강력하고 tabular 데이터에서 자주 우수 |

평가지표:
- **MAE** (평균 절대 오차) → 낮을수록 좋음
- **RMSE** (큰 오차에 더 민감한 지표) → 낮을수록 좋음

---

## 3) Quick Start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\train_baseline.py
```

If you have raw NYC-style data (`data/raw.csv`):
```powershell
python src\prepare_data.py
python src\train_baseline.py
```

---

## 4) Project Structure

- `data/` raw/processed dataset
- `src/prepare_data.py` raw -> train 전처리
- `src/train_baseline.py` 모델 학습/비교 + best model 저장
- `src/train_with_cv.py` TimeSeriesSplit 교차검증
- `models/` best model (`best_model.pkl`) 및 feature metadata
- `reports/` 결과 리포트(csv/png)
- `app.py` Streamlit dashboard

---

## 5) Input Schema (입력 데이터)

`data/train.csv` minimum columns:

| Column | Type | Description |
|---|---|---|
| pickup_datetime | datetime string | 예: `2024-01-01 09:00:00` |
| trip_count | numeric/int | 해당 시점 수요(타겟) |

---

## 6) Output Files (산출물)

| File | Description |
|---|---|
| `reports/model_comparison.csv` | 모델별 MAE/RMSE 비교표 |
| `reports/rmse_comparison.png` | RMSE 막대그래프 |
| `reports/prediction_preview.png` | 실제값 vs 예측값 시각화 |
| `reports/cv_metrics.csv` | TimeSeriesSplit fold별 성능 |
| `models/best_model.pkl` | best model 저장 파일 |
| `models/feature_order.json` | 입력 feature 순서/모델 메타 |

---

## 7) Dashboard

```powershell
streamlit run app.py
```

Dashboard includes:
- Model comparison table
- RMSE chart
- Prediction preview chart
- Quick inference form (hour/dayofweek/month/day)

---

## 8) Key Code Snippets (주요 코드)

### 8.1 Feature Engineering
```python
df["hour"] = df["pickup_datetime"].dt.hour
df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
df["month"] = df["pickup_datetime"].dt.month
df["day"] = df["pickup_datetime"].dt.day
```

### 8.2 Compared Models
```python
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
```

### 8.3 Time-based Split (no shuffle)
```python
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
```

---

## 9) Cross Validation

```powershell
python src\train_with_cv.py
```

Uses `TimeSeriesSplit(n_splits=5)` for more realistic time-series evaluation.

# taxi-demand-forecast

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![CI](https://github.com/zeroziba9-hash/taxi-demand-forecast-/actions/workflows/ci.yml/badge.svg)

NYC Taxi 수요 예측 프로젝트입니다.  
실무형 **시계열 수요 예측(time-series demand forecasting)** 프로젝트로, ML baseline 비교 + 대시보드를 포함합니다.

> **NYC Taxi란?**  
> 뉴욕시 택시 운행 데이터를 의미합니다. 보통 승차 시각, 위치, 요금, 건수 등의 정보가 포함되며,
> 시간대별 수요 예측 연습용으로 많이 사용되는 대표 공개 데이터입니다.

---

## 1) What this project does (프로젝트 목적)

- 택시 수요(`trip_count`)를 시간 기준으로 예측합니다.
- 두 모델을 비교합니다:
  - **RandomForestRegressor**
  - **XGBoostRegressor**
- 결과를 파일(csv/png)로 저장하고, Streamlit 대시보드에서 확인합니다.
- 학습된 best model을 API로 서빙할 수 있습니다.

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
- `src/predict_api.py` FastAPI 기반 추론 API
- `models/` best model (`best_model.pkl`) 및 feature metadata
- `reports/` 결과 리포트(csv/png, experiment log)
- `docs/architecture.md` 아키텍처 다이어그램
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
| `reports/experiment_log.md` | 실험 이력 기록 |
| `models/best_model.pkl` | best model 저장 파일 |
| `models/feature_order.json` | 입력 feature 순서/모델 메타 |

### 산출물 스크린샷

#### Model Comparison Table
두 모델(RandomForest, XGBoost)의 최종 성능을 한눈에 비교한 표입니다.  
MAE/RMSE가 낮을수록 성능이 좋고, 현재는 XGBoost가 근소하게 우세합니다.

![model-comparison-table](docs/screenshots/model_comparison_table.png)

#### CV Metrics Table
시간순 교차검증(TimeSeriesSplit) 결과입니다.  
fold별 점수를 통해 모델이 특정 구간에서만 잘 맞는지, 전반적으로 안정적인지 확인할 수 있습니다.

![cv-metrics-table](docs/screenshots/cv_metrics_table.png)

#### RMSE Comparison Chart
모델별 RMSE를 막대그래프로 시각화한 결과입니다.  
숫자 표보다 빠르게 우열을 확인할 때 유용합니다.

![rmse-comparison](docs/screenshots/rmse_comparison.png)

#### Prediction Preview Chart
실제값(actual)과 예측값(pred)의 흐름을 같이 보여주는 그래프입니다.  
두 선이 비슷하게 움직일수록 예측이 잘 된다는 의미입니다.

![prediction-preview](docs/screenshots/prediction_preview.png)

---

## 7) Dashboard (대시보드)

```powershell
streamlit run app.py
```

대시보드에서 확인할 수 있는 항목:
- 모델 비교 테이블 (Model comparison table)
- RMSE 비교 차트
- 예측값 미리보기 차트 (Prediction preview)
- 빠른 추론 폼 (hour/dayofweek/month/day)

---

## 8) Prediction API (면접관 포인트)

```powershell
uvicorn src.predict_api:app --reload
```

- Health check: `GET /health`
- Inference: `POST /predict`

Example request:
```json
{
  "hour": 9,
  "dayofweek": 1,
  "month": 3,
  "day": 9
}
```

Example response:
```json
{
  "model": "XGBoost",
  "features": {
    "hour": 9,
    "dayofweek": 1,
    "month": 3,
    "day": 9
  },
  "predicted_trip_count": 126.4832
}
```

---

## 9) Architecture

- Diagram: `docs/architecture.md`
- 학습/평가/서빙(대시보드+API) 흐름 분리
- 모델 아티팩트 재사용 구조

---

## 10) Key Code Snippets (주요 코드)

### 10.1 Feature Engineering
```python
df["hour"] = df["pickup_datetime"].dt.hour
df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
df["month"] = df["pickup_datetime"].dt.month
df["day"] = df["pickup_datetime"].dt.day
```

### 10.2 Compared Models
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

### 10.3 Time-based Split (no shuffle)
```python
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
```

---

## 11) Cross Validation

```powershell
python src\train_with_cv.py
```

Uses `TimeSeriesSplit(n_splits=5)` for more realistic time-series evaluation.

---

## 12) CI (자동 검증)

- Workflow: `.github/workflows/ci.yml`
- Push/PR 시 자동으로 아래를 검증합니다:
  - 의존성 설치
  - 샘플 데이터 생성
  - 학습 스크립트 실행
  - 산출물 파일 생성 확인
  - API import smoke check
  - pytest API 기본 테스트 실행

### Local Test
```powershell
pytest -q
```

---

## 13) Business Impact / Limitations

### Business Impact
- 피크 시간대 수요 예측으로 차량 배치 최적화 가능
- 공급 부족 시간대 사전 대응(대기시간/취소율 감소 기대)
- 운영팀이 대시보드 기반으로 빠른 의사결정 가능

### Current Limitations
- 현재는 샘플 기반 feature(시간 변수 중심)만 사용
- 날씨/공휴일/이벤트 feature 미반영
- 하이퍼파라미터 탐색 범위 제한

### Next Step
- lag feature(직전 1시간/24시간) 추가
- 외부 변수(weather/holiday) 결합
- API 배포 + 모니터링(요청량/지연시간) 추가

# Experiment Log

## 2026-03-09 Baseline v1
- Data rows: 2,880 (sample)
- Features: hour, dayofweek, month, day
- Split: time-based 80/20
- Models: RandomForest, XGBoost
- Result:
  - RandomForest RMSE: 12.9000
  - XGBoost RMSE: 12.8685
- Selected best model: XGBoost

## 2026-03-09 CV v1
- Validation: TimeSeriesSplit(n_splits=5)
- Output: `reports/cv_metrics.csv`
- Purpose: fold별 편차 확인 및 일반화 성능 점검

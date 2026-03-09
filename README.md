# taxi-demand-forecast

NYC Taxi demand forecasting project (ML baseline first, then DL comparison).

## Quick Start (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\train_baseline.py
```

If you have a raw NYC-style file (`data/raw.csv`), run:
```powershell
python src\prepare_data.py
python src\train_baseline.py
```

## Project Structure
- `data/` raw and processed datasets
- `notebooks/` exploration notebooks
- `src/` training/inference scripts
- `models/` saved models
- `reports/` evaluation outputs

## Expected Input
Put dataset at:
- `data/train.csv`

Expected columns:
- `pickup_datetime` (datetime string)
- `trip_count` (target)

You can rename columns in `src/train_baseline.py` if needed.

## Current Pipeline
- Time feature engineering (`hour`, `dayofweek`, `month`, `day`)
- Model comparison: RandomForest vs XGBoost
- Metrics export: `reports/model_comparison.csv`
- Visualization export: `reports/rmse_comparison.png`, `reports/prediction_preview.png`

## Dashboard (Streamlit)
```powershell
streamlit run app.py
```

## Time-series Cross Validation
```powershell
python src\train_with_cv.py
```
Output:
- `reports/cv_metrics.csv`

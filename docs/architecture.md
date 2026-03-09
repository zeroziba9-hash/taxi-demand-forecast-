# Architecture

```mermaid
flowchart LR
    A[data/raw.csv] --> B[src/prepare_data.py]
    B --> C[data/train.csv]
    C --> D[src/train_baseline.py]
    D --> E[models/best_model.pkl]
    D --> F[reports/model_comparison.csv]
    D --> G[reports/rmse_comparison.png]
    D --> H[reports/prediction_preview.png]
    C --> I[src/train_with_cv.py]
    I --> J[reports/cv_metrics.csv]
    E --> K[src/predict_api.py]
    E --> L[app.py (Streamlit)]
    F --> L
    G --> L
    H --> L
    J --> L
```

## Notes
- 학습 파이프라인과 추론 경로를 분리했습니다.
- 모델 아티팩트(`best_model.pkl`)를 API와 대시보드가 공통 사용합니다.
- CV 결과(`cv_metrics.csv`)를 별도 산출해 시계열 일반화 성능을 확인합니다.

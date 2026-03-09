from fastapi.testclient import TestClient
from src.predict_api import app

client = TestClient(app)


def test_health():
    res = client.get('/health')
    assert res.status_code == 200
    assert res.json().get('status') == 'ok'


def test_predict_validation_error_for_invalid_hour():
    payload = {
        'hour': 99,
        'dayofweek': 1,
        'month': 3,
        'day': 9,
    }
    res = client.post('/predict', json=payload)
    # validation should fail before model inference
    assert res.status_code == 422

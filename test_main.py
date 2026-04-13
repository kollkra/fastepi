from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict():
    data = {
        "make": "Toyota", "model": "Camry", "year": 2020, "style": "Sedan",
        "distance": 50000, "engine_capacity": 2.5, "fuel_type": "Petrol", "transmission": "Automatic"
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "predicted_price" in response.json()

def test_history():
    response = client.get("/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_health():
    response = client.get("/docs")
    assert response.status_code == 200

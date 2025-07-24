from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_healthcheck():
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

def test_clear_vectors():
    response = client.post("/clear_all_vectors")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_clear_memory():
    response = client.post("/clear-memory")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_conversation_history():
    response = client.get("/conversation-history")
    assert response.status_code == 200
    assert "history" in response.json()
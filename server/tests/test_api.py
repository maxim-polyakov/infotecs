from fastapi.testclient import TestClient

from ueba_server.api import app


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_demo_endpoint(tmp_path) -> None:
    client = TestClient(app)

    response = client.post("/api/demo", json={"output_dir": str(tmp_path / "demo")})

    assert response.status_code == 200
    payload = response.json()
    assert payload["detected_count"] == 3
    assert len(payload["reports"]) == 3


def test_train_endpoint_returns_400_for_too_few_samples(tmp_path) -> None:
    client = TestClient(app)
    data_path = tmp_path / "raw.csv"
    data_path.write_text("cpu_percent,memory_percent,connection_count\n10,40,5\n", encoding="utf-8")

    response = client.post("/api/train", json={"data_path": str(data_path), "model_dir": str(tmp_path / "model")})

    assert response.status_code == 400
    assert "At least two training samples" in response.json()["detail"]

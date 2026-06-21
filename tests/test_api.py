"""
API Integration Tests
=====================
Tests the FastAPI endpoints using httpx's TestClient (no server needed).

Run:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v -m "not db"   # skip DB-dependent tests
"""

import sys
import pytest
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient
    from src.api.main import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI / httpx not installed — pip install fastapi httpx",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def valid_sensor_reading():
    """A valid sensor reading payload."""
    return {
        "air_temp_K":            298.1,
        "process_temp_K":        309.5,
        "rotational_speed_rpm":  1498.0,
        "torque_Nm":             34.2,
        "tool_wear_min":         150.0,
    }


@pytest.fixture
def high_risk_reading():
    """Sensor reading likely to trigger a degraded/critical prediction."""
    return {
        "air_temp_K":            300.0,
        "process_temp_K":        315.0,
        "rotational_speed_rpm":  1800.0,
        "torque_Nm":             75.0,
        "tool_wear_min":         240.0,
    }


# ── Info endpoints ────────────────────────────────────────────────────────────

class TestInfoEndpoints:

    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_contains_name(self, client):
        r = client.get("/")
        assert "CBM" in r.json()["name"]

    def test_root_lists_endpoints(self, client):
        r = client.get("/")
        assert "endpoints" in r.json()

    def test_health_check_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_check_has_status(self, client):
        r = client.get("/health")
        data = r.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")

    def test_health_check_has_timestamp(self, client):
        r = client.get("/health")
        assert "timestamp" in r.json()

    def test_docs_accessible(self, client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_openapi_schema(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert schema["info"]["title"] == "CBM Health Monitoring API"


# ── Prediction endpoint ───────────────────────────────────────────────────────

class TestPredictEndpoint:

    def test_predict_valid_reading(self, client, valid_sensor_reading):
        r = client.post("/predict", json=valid_sensor_reading)
        assert r.status_code == 200

    def test_predict_response_schema(self, client, valid_sensor_reading):
        r = client.post("/predict", json=valid_sensor_reading)
        data = r.json()
        expected_keys = {
            "fault_type", "confidence", "health_score",
            "health_status", "recommendation", "power_W",
            "temp_diff_K", "timestamp",
        }
        assert expected_keys.issubset(data.keys()), \
            f"Missing keys: {expected_keys - data.keys()}"

    def test_predict_health_score_in_range(self, client, valid_sensor_reading):
        r = client.post("/predict", json=valid_sensor_reading)
        hs = r.json()["health_score"]
        assert 0 <= hs <= 100, f"health_score={hs} out of range"

    def test_predict_confidence_in_range(self, client, valid_sensor_reading):
        r = client.post("/predict", json=valid_sensor_reading)
        conf = r.json()["confidence"]
        assert 0 <= conf <= 1, f"confidence={conf} out of range"

    def test_predict_status_is_valid(self, client, valid_sensor_reading):
        r = client.post("/predict", json=valid_sensor_reading)
        assert r.json()["health_status"] in ("Good", "Warning", "Degraded", "Critical")

    def test_predict_computes_power(self, client, valid_sensor_reading):
        r = client.post("/predict", json=valid_sensor_reading)
        data = r.json()
        expected_power = (valid_sensor_reading["torque_Nm"] *
                          valid_sensor_reading["rotational_speed_rpm"] *
                          2 * 3.14159 / 60)
        assert abs(data["power_W"] - expected_power) < 5

    def test_predict_computes_temp_diff(self, client, valid_sensor_reading):
        r = client.post("/predict", json=valid_sensor_reading)
        expected = valid_sensor_reading["process_temp_K"] - valid_sensor_reading["air_temp_K"]
        assert abs(r.json()["temp_diff_K"] - expected) < 0.01

    def test_predict_high_wear_lower_score(self, client, valid_sensor_reading, high_risk_reading):
        r_good = client.post("/predict", json=valid_sensor_reading)
        r_risk = client.post("/predict", json=high_risk_reading)
        assert r_risk.json()["health_score"] <= r_good.json()["health_score"], \
            "High-risk reading should produce equal or lower health score"

    def test_predict_has_recommendation(self, client, valid_sensor_reading):
        r = client.post("/predict", json=valid_sensor_reading)
        rec = r.json()["recommendation"]
        assert isinstance(rec, str) and len(rec) > 5

    def test_predict_missing_field_422(self, client):
        """Missing required field should return 422 Unprocessable Entity."""
        r = client.post("/predict", json={"air_temp_K": 298.1})
        assert r.status_code == 422

    def test_predict_out_of_range_422(self, client, valid_sensor_reading):
        """Air temp below 280 K should fail validation."""
        bad = {**valid_sensor_reading, "air_temp_K": 100.0}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_string_field_422(self, client, valid_sensor_reading):
        """Non-numeric field value should fail validation."""
        bad = {**valid_sensor_reading, "torque_Nm": "forty"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


# ── Batch prediction ──────────────────────────────────────────────────────────

class TestBatchPredict:

    def test_batch_single_item(self, client, valid_sensor_reading):
        r = client.post("/predict/batch", json=[valid_sensor_reading])
        assert r.status_code == 200
        assert isinstance(r.json(), list)
        assert len(r.json()) == 1

    def test_batch_multiple_items(self, client, valid_sensor_reading):
        payload = [valid_sensor_reading] * 5
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        assert len(r.json()) == 5

    def test_batch_all_scores_in_range(self, client, valid_sensor_reading):
        payload = [valid_sensor_reading] * 3
        r = client.post("/predict/batch", json=payload)
        for item in r.json():
            assert 0 <= item["health_score"] <= 100

    def test_batch_too_large_400(self, client, valid_sensor_reading):
        """More than 500 items should return 400."""
        payload = [valid_sensor_reading] * 501
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 400

    def test_batch_empty_list(self, client):
        r = client.post("/predict/batch", json=[])
        assert r.status_code == 200
        assert r.json() == []


# ── Assets endpoints ──────────────────────────────────────────────────────────

@pytest.mark.db
class TestAssetsEndpoints:
    """Tests that require a populated database."""

    def test_assets_list_200(self, client):
        r = client.get("/assets")
        # 200 if DB exists, 503 if not — both are valid in test environment
        assert r.status_code in (200, 503)

    def test_assets_unknown_404(self, client):
        r = client.get("/assets/NONEXISTENT-999")
        assert r.status_code in (404, 503)

    def test_fleet_summary_200(self, client):
        r = client.get("/fleet/summary")
        assert r.status_code in (200, 503)

    def test_fleet_summary_schema(self, client):
        r = client.get("/fleet/summary")
        if r.status_code == 200:
            data = r.json()
            assert "total_assets" in data
            assert "avg_health_score" in data
            assert "fleet_availability" in data
            assert 0 <= data["fleet_availability"] <= 100

    def test_recommendations_200(self, client):
        r = client.get("/recommendations")
        assert r.status_code in (200, 503)

    def test_recommendations_priority_filter(self, client):
        r = client.get("/recommendations?priority=High")
        assert r.status_code in (200, 503)

    def test_recommendations_invalid_limit(self, client):
        r = client.get("/recommendations?limit=9999")
        assert r.status_code == 422

"""
Integration tests for src/api/main.py via FastAPI TestClient.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_model_loaded(self, client):
        r = client.get("/health")
        assert r.json()["model_loaded"] is True


# ---------------------------------------------------------------------------
# /gpus
# ---------------------------------------------------------------------------

class TestListGpus:
    def test_returns_gpu_list(self, client):
        r = client.get("/gpus")
        assert r.status_code == 200
        gpus = r.json()["gpus"]
        # The spec DB has 8 required in-scope GPUs; len > 0 would pass even if
        # all but one were dropped.
        assert len(gpus) >= 8

    def test_mi300x_present(self, client):
        r = client.get("/gpus")
        ids = [g["id"] for g in r.json()["gpus"]]
        assert "mi300x" in ids

    def test_gpu_fields_present(self, client):
        r = client.get("/gpus")
        gpu = r.json()["gpus"][0]
        # Verify keys exist AND values are populated with correct types — a dict
        # full of None values would pass a bare "in gpu" key-presence check.
        assert isinstance(gpu["id"], str) and gpu["id"]
        assert gpu["vendor"] in ("amd", "nvidia")
        assert isinstance(gpu["vram_gb"], (int, float)) and gpu["vram_gb"] > 0
        assert isinstance(gpu["in_model_scope"], bool)


# ---------------------------------------------------------------------------
# /models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_returns_model_list(self, client):
        from src.models.predictor import VALID_MODELS
        r = client.get("/models")
        assert r.status_code == 200
        # Compare the complete set — spot-checking 2 models misses regressions
        # that remove other models (e.g. mixtral-8x7b, llama3.1-405b).
        assert set(r.json()["models"]) == VALID_MODELS


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_response_has_throughput(self, client):
        r = client.post("/predict", json={
            "gpu_id": "h100_sxm",
            "model_name": "llama2-70b",
            "scenario": "Offline",
            "accuracy_tier": "99",
            "framework": "tensorrt",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["pred_throughput_tok_per_sec"] >= 0.05 * data["roofline_tput_tok_per_sec"]

    def test_throughput_not_exceeds_roofline(self, client):
        r = client.post("/predict", json={
            "gpu_id": "mi300x",
            "model_name": "llama2-70b",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["pred_throughput_tok_per_sec"] <= data["roofline_tput_tok_per_sec"] + 1e-2

    def test_unknown_gpu_returns_422(self, client):
        r = client.post("/predict", json={
            "gpu_id": "titan_v",
            "model_name": "llama2-70b",
        })
        assert r.status_code == 422

    def test_unknown_model_returns_422(self, client):
        r = client.post("/predict", json={
            "gpu_id": "mi300x",
            "model_name": "gpt5",
        })
        assert r.status_code == 422

    def test_invalid_scenario_returns_422(self, client):
        r = client.post("/predict", json={
            "gpu_id": "mi300x",
            "model_name": "gptj",
            "scenario": "Interactive",
        })
        assert r.status_code == 422

    def test_defaults_filled(self, client):
        r = client.post("/predict", json={
            "gpu_id": "h200_sxm",
            "model_name": "gptj",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["scenario"] == "Offline"
        assert data["accuracy_tier"] == "99"
        assert data["framework"] == "vllm"


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_batch_returns_correct_count(self, client):
        r = client.post("/predict/batch", json=[
            {"gpu_id": "mi300x", "model_name": "llama2-70b"},
            {"gpu_id": "h100_sxm", "model_name": "gptj"},
        ])
        assert r.status_code == 200
        results = r.json()
        assert len(results) == 2
        # Verify identity, not just count — [mi300x, mi300x] would also have len 2.
        assert results[0]["gpu_id"] == "mi300x" and results[0]["model_name"] == "llama2-70b"
        assert results[1]["gpu_id"] == "h100_sxm" and results[1]["model_name"] == "gptj"

    def test_empty_batch_returns_empty(self, client):
        r = client.post("/predict/batch", json=[])
        assert r.status_code == 200
        assert r.json() == []

    def test_batch_over_50_returns_422(self, client):
        batch = [{"gpu_id": "mi300x", "model_name": "gptj"}] * 51
        r = client.post("/predict/batch", json=batch)
        assert r.status_code == 422

    def test_body_too_large_returns_413(self, client):
        # Send 1 MB + 1 byte — the limit_body_size middleware must intercept
        # before Pydantic validation (which would return 422, not 413).
        # The 51-item batch test above only exercises the Pydantic max_length path.
        from src.api.main import _MAX_BODY_BYTES
        r = client.post(
            "/predict/batch",
            content=b"x" * (_MAX_BODY_BYTES + 1),
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 413


# ---------------------------------------------------------------------------
# POST /recommend
# ---------------------------------------------------------------------------

class TestRecommend:
    def test_workload_echoed(self, client):
        r = client.post("/recommend", json={
            "model_name": "mixtral-8x7b",
            "scenario": "Server",
            "accuracy_tier": "base",
            "framework": "rocm_other",
        })
        assert r.status_code == 200
        wl = r.json()["workload"]
        # All four echoed fields must round-trip — previously only model_name and
        # scenario were asserted; accuracy_tier and framework were silently unchecked.
        assert wl["model_name"] == "mixtral-8x7b"
        assert wl["scenario"] == "Server"
        assert wl["accuracy_tier"] == "base"
        assert wl["framework"] == "rocm_other"

    def test_budget_filter_in_recommend(self, client):
        r = client.post("/recommend", json={
            "model_name": "gptj",
            "budget_per_gpu_hr": 1.0,
        })
        assert r.status_code == 200
        candidates = r.json()["frontier"] + r.json()["dominated"]
        # L4 ($0.44) and RTX4090 ($0.39) are both under $1.00/hr and fit gptj
        # (6 GB).  Without this guard the loop below fires zero assertions if
        # all GPUs exceed the budget (e.g. after a pricing update).
        assert len(candidates) > 0, (
            "Expected at least one GPU within $1.00/hr (L4 $0.44, RTX4090 $0.39)"
        )
        # All in-scope GPUs have pricing (validated at startup), so
        # price_per_gpu_hr is never None here — the is-None arm was dead code.
        for gpu in candidates:
            assert gpu["price_per_gpu_hr"] <= 1.0

    def test_unknown_model_returns_422(self, client):
        r = client.post("/recommend", json={"model_name": "invalid_model"})
        assert r.status_code == 422

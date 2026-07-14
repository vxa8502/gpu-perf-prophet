"""Integration tests for src/api/main.py via FastAPI TestClient."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.middleware import RateLimiter

_REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


# /health

class TestHealth:
    def test_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_model_loaded(self, client):
        r = client.get("/health")
        assert r.json()["model_loaded"] is True

    def test_returns_provenance_fields_matching_version_endpoint(self, client):
        # Checked against /version's own response, not re-derived from source files again — TestVersion.test_returns_provenance_fields already independently verifies /version against the raw files; this test only confirms /health doesn't drift from it.
        health_body = client.get("/health").json()
        version_body = client.get("/version").json()
        for field in ("model_artifact_version", "model_artifact_sha256", "pricing_snapshot_date", "gpu_spec_db_version"):
            assert health_body[field] == version_body[field]

    def test_degrades_gracefully_without_provenance_when_model_not_loaded(self, client, monkeypatch):
        # /version raises 503 when _predictor is None; /health must not, since it exists to report exactly that "not ready yet" state.
        import src.api.main as main_module
        monkeypatch.setattr(main_module, "_predictor", None)
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body == {"status": "ok", "model_loaded": False}


# /gpus

class TestListGpus:
    def test_returns_gpu_list(self, client):
        r = client.get("/gpus")
        assert r.status_code == 200
        gpus = r.json()["gpus"]
        # The spec DB has 8 required in-scope GPUs; len > 0 would pass even if all but one were dropped.
        assert len(gpus) >= 8

    def test_mi300x_present(self, client):
        r = client.get("/gpus")
        ids = [g["id"] for g in r.json()["gpus"]]
        assert "mi300x" in ids

    def test_gpu_fields_present(self, client):
        r = client.get("/gpus")
        gpu = r.json()["gpus"][0]
        # Verify keys exist AND values are populated with correct types — a dict full of None values would pass a bare "in gpu" key-presence check.
        assert isinstance(gpu["id"], str) and gpu["id"]
        assert gpu["vendor"] in ("amd", "nvidia")
        assert isinstance(gpu["vram_gb"], (int, float)) and gpu["vram_gb"] > 0
        assert isinstance(gpu["in_model_scope"], bool)


# /models

class TestListModels:
    def test_returns_model_list(self, client):
        from src.models.predictor import VALID_MODELS
        r = client.get("/models")
        assert r.status_code == 200
        # Compare the complete set — spot-checking 2 models misses regressions that remove other models (e.g. mixtral-8x7b, llama3.1-405b).
        assert set(r.json()["models"]) == VALID_MODELS


# GET /version

class TestVersion:
    def test_returns_provenance_fields(self, client):
        # Each expected value is independently recomputed from the raw source file — not imported from src.models.predictor/src.data.gpu_spec_db/src.recommend.recommender — so this fails if any of those computations silently drift, not just if the field goes missing.
        r = client.get("/version")
        assert r.status_code == 200
        body = r.json()

        model_bytes = (_REPO_ROOT / "data" / "models" / "prophet_v1.json").read_bytes()
        assert body["model_artifact_sha256"] == hashlib.sha256(model_bytes).hexdigest()

        meta = json.loads((_REPO_ROOT / "data" / "models" / "feature_metadata.json").read_text())
        assert body["model_artifact_version"] == meta["model_version"]

        specs = yaml.safe_load((_REPO_ROOT / "data" / "gpu_specs.yaml").read_text())
        assert body["gpu_spec_db_version"] == str(specs["schema_version"])

        pricing = yaml.safe_load((_REPO_ROOT / "data" / "pricing.yaml").read_text())
        assert body["pricing_snapshot_date"] == pricing["source_date"]


# Observability middleware (request-id, access log, rate limit)

class TestObservability:
    def test_request_id_header_present_and_unique(self, client):
        r1 = client.get("/health")
        r2 = client.get("/health")
        assert "x-request-id" in r1.headers
        assert r1.headers["x-request-id"] != r2.headers["x-request-id"]

    def test_meta_block_on_predict(self, client):
        r = client.post("/predict", json={
            "gpu_id": "h100_sxm",
            "model_name": "llama2-70b",
            "accuracy_tier": "99",
        })
        assert r.status_code == 200
        meta = r.json()["meta"]
        assert meta["request_id"] == r.headers["x-request-id"]
        assert meta["model_artifact_sha256"]
        assert meta["gpu_spec_db_version"]

    def test_meta_block_on_recommend(self, client):
        r = client.post("/recommend", json={"model_name": "llama2-70b", "accuracy_tier": "99"})
        assert r.status_code == 200
        meta = r.json()["meta"]
        assert meta["request_id"] == r.headers["x-request-id"]

    def test_meta_matches_version_provenance(self, client):
        # /version and every response's meta block draw their 4 provenance fields from one shared helper (_provenance_fields) rather than two independently-maintained copies.
        version = client.get("/version").json()
        meta = client.post("/predict", json={
            "gpu_id": "h100_sxm",
            "model_name": "llama2-70b",
            "accuracy_tier": "99",
        }).json()["meta"]
        for key in version:
            assert meta[key] == version[key], f"{key}: meta={meta[key]!r} version={version[key]!r}"

    def test_meta_present_on_value_error_422(self, client):
        # A ValueError raised from inside predict()/recommend() (after the predictor is already loaded) becomes a 422 via `except ValueError -> HTTPException(422, ...)` — this response must still carry meta, unlike a Pydantic-level RequestValidationError (a different exception type, not covered).
        r = client.post("/predict", json={
            "gpu_id": "not-a-real-gpu",
            "model_name": "llama2-70b",
            "accuracy_tier": "99",
        })
        assert r.status_code == 422
        body = r.json()
        assert body["meta"]["request_id"] == r.headers["x-request-id"]
        assert body["meta"]["model_artifact_sha256"]

    def test_access_log_line_emitted(self, client, caplog):
        # One structured JSON log line per request is expected; this asserts log.info() is actually called with the right shape, independent of whether logging.basicConfig gives it a visible handler under uvicorn.
        import json as _json
        with caplog.at_level("INFO", logger="gpp.access"):
            r = client.get("/health")
        lines = [rec.message for rec in caplog.records if rec.name == "gpp.access"]
        assert len(lines) == 1
        payload = _json.loads(lines[0])
        assert payload["request_id"] == r.headers["x-request-id"]
        assert payload["route"] == "/health"
        assert payload["status"] == 200
        assert "latency_ms" in payload

    def test_rate_limit_returns_429_above_burst(self, client, monkeypatch):
        # RATE_LIMIT_RPM=60 tokens, refilled at 1/s; a tight burst of 61 requests on a fresh bucket must trip the limiter on the 61st. `_rate_limiter` is a module-level singleton shared by every request, so monkeypatching a fresh instance keeps this burst from draining the bucket other tests rely on.
        import src.api.main as main_module
        monkeypatch.setattr(main_module, "_rate_limiter", RateLimiter())
        statuses = [client.get("/health").status_code for _ in range(61)]
        assert statuses.count(429) >= 1


# POST /predict

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

    def test_has_training_data_survives_http_round_trip(self, client):
        # FastAPI's response_model silently drops any dict key not declared on the Pydantic schema (it would vanish over HTTP without raising) — assert the actual JSON, not the Python dict predictor.predict() returns.
        r = client.post("/predict", json={"gpu_id": "mi300x", "model_name": "gptj"})
        assert r.status_code == 200
        assert r.json()["has_training_data"] is True
        # mi300x has real rows but sits under this project's 100-row-per-GPU floor.
        assert r.json()["training_data_tier"] == "below_floor"

        r = client.post("/predict", json={"gpu_id": "rtx4090", "model_name": "gptj"})
        assert r.status_code == 200
        assert r.json()["has_training_data"] is False
        assert r.json()["training_data_tier"] == "none"

        r = client.post("/predict", json={"gpu_id": "h100_sxm", "model_name": "gptj"})
        assert r.status_code == 200
        assert r.json()["has_training_data"] is True
        assert r.json()["training_data_tier"] == "sufficient"

    def test_memory_fit_fields_survive_http_round_trip(self, client):
        r = client.post("/predict", json={"gpu_id": "mi300x", "model_name": "llama2-70b"})
        assert r.status_code == 200
        data = r.json()
        for key in ("memory_fit_verdict", "kv_cache_gb", "memory_total_gb", "vram_utilization"):
            assert key in data, f"{key} missing from HTTP response"
        assert data["memory_fit_verdict"] in ("fits", "tight", "does_not_fit")

    def test_batch_size_out_of_range_returns_422(self, client):
        r = client.post("/predict", json={
            "gpu_id": "mi300x", "model_name": "gptj", "batch_size": 0,
        })
        assert r.status_code == 422

    def test_input_tokens_out_of_range_returns_422(self, client):
        r = client.post("/predict", json={
            "gpu_id": "mi300x", "model_name": "gptj", "input_tokens": 100_000,
        })
        assert r.status_code == 422

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

    def test_unsupported_precision_returns_422(self, client):
        # a100_sxm_80gb (Ampere) has no native FP8; accuracy_tier="99" selects fp8 and must return a structured 422, not a 200 with a silently substituted (and physically inconsistent) fp16-ceiling prediction.
        r = client.post("/predict", json={
            "gpu_id": "a100_sxm_80gb",
            "model_name": "gptj",
            "accuracy_tier": "99",
        })
        assert r.status_code == 422
        assert "fp8" in r.json()["detail"]

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

    def test_memory_fit_defaults_filled(self, client):
        # `memory_fit_verdict in (three valid strings)` is true unconditionally and wouldn't prove DEFAULT_* was actually applied, so instead prove it two ways: (1) omitting the fields must byte-for-byte match passing DEFAULT_BATCH_SIZE/DEFAULT_INPUT_TOKENS/DEFAULT_OUTPUT_TOKENS explicitly; (2) a different batch/token combo must change the numbers, ruling out a stub that ignores the fields but coincidentally satisfies (1).
        from src.features.build_features import (
            DEFAULT_BATCH_SIZE, DEFAULT_INPUT_TOKENS, DEFAULT_OUTPUT_TOKENS,
        )

        r_omitted = client.post("/predict", json={"gpu_id": "h200_sxm", "model_name": "gptj"})
        r_explicit = client.post("/predict", json={
            "gpu_id": "h200_sxm", "model_name": "gptj",
            "batch_size": DEFAULT_BATCH_SIZE,
            "input_tokens": DEFAULT_INPUT_TOKENS,
            "output_tokens": DEFAULT_OUTPUT_TOKENS,
        })
        assert r_omitted.status_code == 200
        assert r_explicit.status_code == 200
        for key in ("memory_fit_verdict", "kv_cache_gb", "memory_total_gb", "vram_utilization"):
            assert r_omitted.json()[key] == r_explicit.json()[key], (
                f"{key}: omitting batch/token fields ({r_omitted.json()[key]!r}) diverged from "
                f"passing DEFAULT_* explicitly ({r_explicit.json()[key]!r})"
            )

        r_different = client.post("/predict", json={
            "gpu_id": "h200_sxm", "model_name": "gptj",
            "batch_size": 1, "input_tokens": 64, "output_tokens": 1,
        })
        assert r_different.json()["kv_cache_gb"] != r_omitted.json()["kv_cache_gb"], (
            "kv_cache_gb didn't change with a different batch/token combo — "
            "batch_size/input_tokens/output_tokens may not actually be wired in"
        )


# POST /predict/batch

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
        # Send 1 MB + 1 byte — the limit_body_size middleware must intercept before Pydantic validation (which would return 422, not 413); the 51-item batch test above only exercises the Pydantic max_length path.
        from src.api.main import _MAX_BODY_BYTES
        r = client.post(
            "/predict/batch",
            content=b"x" * (_MAX_BODY_BYTES + 1),
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 413


# POST /recommend

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
        # All four echoed fields must round-trip — previously only model_name and scenario were asserted; accuracy_tier and framework were silently unchecked.
        assert wl["model_name"] == "mixtral-8x7b"
        assert wl["scenario"] == "Server"
        assert wl["accuracy_tier"] == "base"
        assert wl["framework"] == "rocm_other"

    def test_ranking_objective_echoed_and_defaults_to_tokens_per_dollar(self, client):
        r = client.post("/recommend", json={"model_name": "gptj"})
        assert r.status_code == 200
        assert r.json()["workload"]["ranking_objective"] == "tokens_per_dollar"

    def test_ranking_objective_override_echoed(self, client):
        r = client.post("/recommend", json={
            "model_name": "gptj", "ranking_objective": "tokens_per_watt",
        })
        assert r.status_code == 200
        assert r.json()["workload"]["ranking_objective"] == "tokens_per_watt"

    def test_invalid_ranking_objective_rejected_before_recommender(self, client):
        # Pydantic's Literal type rejects this before GpuRecommender.recommend() is ever called — a clean 422, same convention as scenario/accuracy_tier.
        r = client.post("/recommend", json={
            "model_name": "gptj", "ranking_objective": "most_tokens_ever",
        })
        assert r.status_code == 422

    def test_watts_tokens_per_watt_cost_per_million_present_over_http(self, client):
        # watts/tokens_per_watt/cost_per_million_tokens are asserted over a real HTTP round-trip (same silent-field-drop risk as has_training_data) and are checked against an independent /predict call for mi300x (tdp_w=750) rather than this response's own throughput/price fields, because mutation testing found that reusing the response's own throughput to build "expected" fails to catch throughput being wired to the wrong value (e.g. roofline_tput_tok_per_sec) since both fields move together and the test stays green.
        r = client.post("/recommend", json={"model_name": "gptj", "accuracy_tier": "99"})
        assert r.status_code == 200
        candidates = r.json()["frontier"] + r.json()["dominated"]
        assert len(candidates) > 0
        for gpu in candidates:
            assert gpu["watts"] is not None and gpu["watts"] > 0
            assert gpu["cost_per_million_tokens"] > 0

        mi300x = next(g for g in candidates if g["gpu_id"] == "mi300x")
        assert mi300x["watts"] == 750
        standalone = client.post(
            "/predict", json={"gpu_id": "mi300x", "model_name": "gptj", "accuracy_tier": "99"},
        ).json()
        real_tput = standalone["pred_throughput_tok_per_sec"]
        assert mi300x["throughput"] == pytest.approx(real_tput)
        assert mi300x["tokens_per_watt"] == pytest.approx(real_tput / 750)

    def test_memory_fit_inputs_echoed_in_workload(self, client):
        r = client.post("/recommend", json={
            "model_name": "gptj",
            "batch_size": 8,
            "input_tokens": 512,
            "output_tokens": 128,
        })
        assert r.status_code == 200
        wl = r.json()["workload"]
        assert wl["batch_size"] == 8
        assert wl["input_tokens"] == 512
        assert wl["output_tokens"] == 128

    def test_budget_filter_in_recommend(self, client):
        # Light batch/context override keeps the KV cache small enough that L4/RTX4090 (24 GB) still fit gptj — the default batch=32/2048-token assumption needs ~25 GB and would exclude both.
        r = client.post("/recommend", json={
            "model_name": "gptj",
            "batch_size": 1,
            "input_tokens": 128,
            "output_tokens": 64,
            "budget_per_gpu_hr": 1.0,
        })
        assert r.status_code == 200
        candidates = r.json()["frontier"] + r.json()["dominated"]
        # L4 ($0.44) and RTX4090 ($0.39) are both under $1.00/hr and fit gptj (6 GB) — without this guard the loop below fires zero assertions if all GPUs exceed the budget (e.g. after a pricing update).
        assert len(candidates) > 0, (
            "Expected at least one GPU within $1.00/hr (L4 $0.44, RTX4090 $0.39)"
        )
        # All in-scope GPUs have pricing (validated at startup), so price_per_gpu_hr is never None here — the is-None arm was dead code.
        for gpu in candidates:
            assert gpu["price_per_gpu_hr"] <= 1.0

    def test_cheap_budget_top_pick_flagged_over_http(self, client):
        # Same silent-field-drop risk as /predict, plus this is the actual scenario that motivated the feature: a tight-budget query's top pick is a GPU with zero real training rows.
        r = client.post("/recommend", json={
            "model_name": "gptj",
            "batch_size": 1,
            "input_tokens": 128,
            "output_tokens": 64,
            "budget_per_gpu_hr": 1.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["frontier"], "Expected at least one candidate under $1.00/hr"
        assert body["frontier"][0]["gpu_id"] == "rtx4090"
        assert body["frontier"][0]["has_training_data"] is False
        assert body["frontier"][0]["training_data_tier"] == "none"
        for gpu in body["frontier"] + body["dominated"] + body["filtered"]:
            assert "has_training_data" in gpu
            assert "training_data_tier" in gpu

    def test_unknown_model_returns_422(self, client):
        r = client.post("/recommend", json={"model_name": "invalid_model"})
        assert r.status_code == 422

    def test_unsupported_precision_gpu_filtered_not_500(self, client):
        # Unsupported-precision handling in the recommend() multi-GPU path: a100_sxm_80gb (no native FP8) must be excluded with a reason, not cause a 500 by reaching predict_batch() (which now raises for this case).
        r = client.post("/recommend", json={
            "model_name": "gptj",
            "accuracy_tier": "99",
        })
        assert r.status_code == 200
        body = r.json()
        candidate_ids = {g["gpu_id"] for g in body["frontier"] + body["dominated"]}
        assert "a100_sxm_80gb" not in candidate_ids
        entry = next(f for f in body["filtered"] if f["gpu_id"] == "a100_sxm_80gb")
        assert "fp8" in entry["reject_reason"]

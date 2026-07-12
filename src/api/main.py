"""
GPU Perf Prophet — FastAPI application.

Routes
------
GET  /health                   liveness probe
GET  /gpus                     list all GPU ids in the spec DB
GET  /models                   list all supported LLM model names
POST /predict                  predict throughput for one (GPU, workload) pair
POST /predict/batch            vectorised prediction for up to 50 pairs
POST /recommend                Pareto GPU recommendation for a workload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import Field

from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
)
from src.data.gpu_spec_db import load_specs
from src.models.predictor import GpuPredictor, VALID_MODELS
from src.recommend.recommender import GpuRecommender

# Hard cap on raw request body size — prevents parsing-based DoS before any
# Pydantic validation runs.  50 PredictRequest objects at ~200 bytes each ≈
# 10 KB; 1 MB is a generous ceiling that still blocks runaway bodies.
_MAX_BODY_BYTES = 1 * 1024 * 1024  # 1 MB

log = logging.getLogger(__name__)

_predictor: GpuPredictor | None = None
_recommender: GpuRecommender | None = None
_gpu_list: list[dict] | None = None
_model_list: list[str] = sorted(VALID_MODELS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _recommender, _gpu_list
    log.info("Loading model …")
    _predictor = GpuPredictor()
    _recommender = GpuRecommender(_predictor)
    specs = load_specs()
    _gpu_list = [
        {
            "id": s["id"],
            "name": s.get("name", s["id"]),
            "vendor": s.get("vendor"),
            "architecture": s.get("architecture"),
            "vram_gb": s.get("vram_gb"),
            "in_model_scope": s.get("in_model_scope", False),
        }
        for s in specs
    ]
    log.info("Ready.")
    yield
    _predictor = None
    _recommender = None
    _gpu_list = None


app = FastAPI(
    title="GPU Perf Prophet",
    description=(
        "Cross-vendor LLM inference performance forecasting "
        "and workload-aware GPU recommendation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def limit_body_size(request: Request, call_next) -> Response:
    # Fast path: reject if Content-Length header exceeds the limit.
    # A malformed or missing Content-Length header is not an error here —
    # it just means we must check the actual body below.
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > _MAX_BODY_BYTES:
                return Response(status_code=413, content="Request body too large")
        except ValueError:
            return Response(status_code=400, content="Invalid Content-Length header")

    # GET / HEAD / OPTIONS carry no request body — no route handler reads one,
    # so the size limit is irrelevant and buffering is wasted work.
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return await call_next(request)

    # Slow path: consume the actual body so chunked-transfer or lying clients
    # cannot bypass the Content-Length check above.  Without this step, a
    # client sending without Content-Length (chunked encoding) would bypass
    # the limit entirely because uvicorn imposes no default body size cap.
    body = await request.body()
    if len(body) > _MAX_BODY_BYTES:
        return Response(status_code=413, content="Request body too large")

    # Re-inject the consumed body so downstream route handlers can read it.
    async def _receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = _receive
    return await call_next(request)


def _get_predictor() -> GpuPredictor:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predictor


def _get_recommender() -> GpuRecommender:
    if _recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")
    return _recommender


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.get("/gpus")
def list_gpus() -> dict:
    return {"gpus": _gpu_list or []}


@app.get("/models")
def list_models() -> dict:
    return {"models": _model_list}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> dict:
    predictor = _get_predictor()
    try:
        return predictor.predict(
            gpu_id=req.gpu_id,
            model_name=req.model_name,
            scenario=req.scenario,
            accuracy_tier=req.accuracy_tier,
            framework=req.framework,
            batch_size=req.batch_size,
            input_tokens=req.input_tokens,
            output_tokens=req.output_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(
    requests: Annotated[list[PredictRequest], Field(max_length=50)],
) -> list[dict]:
    predictor = _get_predictor()
    try:
        return predictor.predict_batch([r.model_dump() for r in requests])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> dict:
    recommender = _get_recommender()
    try:
        return recommender.recommend(
            model_name=req.model_name,
            scenario=req.scenario,
            accuracy_tier=req.accuracy_tier,
            framework=req.framework,
            batch_size=req.batch_size,
            input_tokens=req.input_tokens,
            output_tokens=req.output_tokens,
            budget_per_gpu_hr=req.budget_per_gpu_hr,
            min_throughput_tok_per_sec=req.min_throughput_tok_per_sec,
            ranking_objective=req.ranking_objective,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

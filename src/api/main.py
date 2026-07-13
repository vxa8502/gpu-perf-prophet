"""GPU Perf Prophet FastAPI app: GET /health, /version, /gpus, /models; POST /predict, /predict/batch, /recommend."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import Field

from src.api.middleware import RateLimiter, log_access, new_request_id
from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
    VersionOut,
)
from src.data.gpu_spec_db import load_specs, spec_db_version
from src.models.predictor import GpuPredictor, VALID_MODELS
from src.recommend.recommender import GpuRecommender

# Hard cap on raw body size to block parsing-based DoS before Pydantic validation runs; 50 requests ≈ 10 KB, so 1 MB is a generous ceiling.
_MAX_BODY_BYTES = 1 * 1024 * 1024  # 1 MB

# Without this, the root logger has no handler and every log.info() call (including the pre-existing "Loading model…" lifespan lines) is silently dropped — uvicorn only configures its own "uvicorn"/"uvicorn.access"/"uvicorn.error" loggers, never root; format="%(message)s" keeps the access log's JSON lines bare and directly log-shippable.
logging.basicConfig(level=logging.INFO, format="%(message)s")

log = logging.getLogger(__name__)

_predictor: GpuPredictor | None = None
_recommender: GpuRecommender | None = None
_gpu_list: list[dict] | None = None
_model_list: list[str] = sorted(VALID_MODELS)
_gpu_spec_db_version: str = "unknown"
_rate_limiter = RateLimiter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _recommender, _gpu_list, _gpu_spec_db_version
    log.info("Loading model …")
    _predictor = GpuPredictor()
    _recommender = GpuRecommender(_predictor)
    specs = load_specs()
    _gpu_spec_db_version = spec_db_version()
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
    # Fast path: reject via Content-Length if present; a missing/malformed header isn't an error, it just means checking the actual body below.
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > _MAX_BODY_BYTES:
                return Response(status_code=413, content="Request body too large")
        except ValueError:
            return Response(status_code=400, content="Invalid Content-Length header")

    # GET/HEAD/OPTIONS carry no request body, so buffering here is wasted work.
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return await call_next(request)

    # Slow path: consume the body so chunked-transfer clients (no Content-Length) can't bypass the check — uvicorn enforces no default body size cap.
    body = await request.body()
    if len(body) > _MAX_BODY_BYTES:
        return Response(status_code=413, content="Request body too large")

    # Re-inject the consumed body so downstream route handlers can read it.
    async def _receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = _receive
    return await call_next(request)


# Registered after limit_body_size (so it wraps it — the outermost middleware runs first on the way in and last on the way out) to keep request-id/rate-limit/access-log applying to every response, including 413s from the body-size guard above.
@app.middleware("http")
async def observability(request: Request, call_next) -> Response:
    request_id = new_request_id()
    request.state.request_id = request_id

    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.allow(client_ip):
        response = Response(status_code=429, content="Rate limit exceeded")
        response.headers["X-Request-ID"] = request_id
        log_access(request_id, request, 429, 0.0)
        return response

    start = time.monotonic()
    response = await call_next(request)
    latency_ms = (time.monotonic() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    log_access(request_id, request, response.status_code, latency_ms)
    return response


def _get_predictor() -> GpuPredictor:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predictor


def _get_recommender() -> GpuRecommender:
    if _recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")
    return _recommender


def _provenance_fields() -> dict:
    """The 4 fields VersionOut and MetaOut share (everything but MetaOut's request_id) — the single source both /version and _build_meta() read from, so a future provenance field only needs adding in one place instead of two independently-maintained copies."""
    predictor = _get_predictor()
    return {
        "model_artifact_version": predictor.model_version,
        "model_artifact_sha256": predictor.model_artifact_sha256,
        "pricing_snapshot_date": _recommender.pricing_source_date if _recommender else None,
        "gpu_spec_db_version": _gpu_spec_db_version,
    }


def _build_meta(request: Request) -> dict:
    return {"request_id": request.state.request_id, **_provenance_fields()}


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    # meta needs a loaded predictor to compute its fields; the 503 "not loaded" case is the one HTTPException path where that data genuinely doesn't exist yet, so it's omitted there rather than faked. Every other HTTPException here (422s from `except ValueError` in predict/predict_batch/recommend) only happens after the predictor is already loaded. Pydantic's own RequestValidationError (e.g. a bad accuracy_tier Literal) is a separate exception type this handler does not cover.
    body: dict = {"detail": exc.detail}
    if _predictor is not None:
        body["meta"] = _build_meta(request)
    return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)


# Routes

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.get("/version", response_model=VersionOut)
def version() -> dict:
    return _provenance_fields()


@app.get("/gpus")
def list_gpus() -> dict:
    return {"gpus": _gpu_list or []}


@app.get("/models")
def list_models() -> dict:
    return {"models": _model_list}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> dict:
    predictor = _get_predictor()
    try:
        result = predictor.predict(
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
    result["meta"] = _build_meta(request)
    return result


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(
    requests: Annotated[list[PredictRequest], Field(max_length=50)],
    request: Request,
) -> list[dict]:
    predictor = _get_predictor()
    try:
        results = predictor.predict_batch([r.model_dump() for r in requests])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    meta = _build_meta(request)
    for result in results:
        result["meta"] = meta
    return results


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, request: Request) -> dict:
    recommender = _get_recommender()
    try:
        result = recommender.recommend(
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
    result["meta"] = _build_meta(request)
    return result

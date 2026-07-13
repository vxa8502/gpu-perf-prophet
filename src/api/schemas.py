"""Pydantic request/response schemas for the GPU Perf Prophet API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


Scenario      = Literal["Offline", "Server"]
AccuracyTier  = Literal["base", "99", "99.9"]
Framework     = Literal["vllm", "tensorrt", "rocm_other", "other"]
# Unlike Scenario/AccuracyTier/Framework, this is entirely server-computed (build_features.memory_fit_verdict), not request-validated, hence Literal not str.
MemoryFitVerdict = Literal["fits", "tight", "does_not_fit"]
# Same shape as MemoryFitVerdict: server-computed from GpuPredictor.training_data_tier, not request input.
TrainingDataTier = Literal["none", "below_floor", "sufficient"]
# The four named ranking scalars; validated against recommender.VALID_RANKING_OBJECTIVES.
RankingObjective = Literal[
    "tokens_per_dollar", "tokens_per_second", "tokens_per_watt", "lowest_cost_per_million_tokens"
]


class MetaOut(BaseModel):
    """Per-request provenance: what code+data actually produced this response, so a caller can tell whether two responses are comparable."""
    request_id:                str
    model_artifact_version:    str
    model_artifact_sha256:     str
    pricing_snapshot_date:     Optional[str]
    gpu_spec_db_version:       str


class VersionOut(BaseModel):
    """GET /version: same provenance fields as MetaOut, without a per-request request_id."""
    model_artifact_version:    str
    model_artifact_sha256:     str
    pricing_snapshot_date:     Optional[str]
    gpu_spec_db_version:       str


class PredictRequest(BaseModel):
    gpu_id:        str          = Field(..., max_length=100, examples=["mi300x"])
    model_name:    str          = Field(..., max_length=100, examples=["llama2-70b"])
    scenario:      Scenario     = "Offline"
    accuracy_tier: AccuracyTier = "99"
    framework:     Framework    = "vllm"
    # KV-cache memory-fit inputs only, not ML features — MLPerf carries no per-row batch/context length, so these are overridable assumptions mirroring build_features.DEFAULT_*.
    batch_size:    int = Field(32,   ge=1,  le=256)
    input_tokens:  int = Field(2048, ge=64, le=8192)
    output_tokens: int = Field(256,  ge=1,  le=4096)


class PredictResponse(BaseModel):
    gpu_id:                        str
    model_name:                    str
    scenario:                      str
    accuracy_tier:                 str
    framework:                     str
    pred_throughput_tok_per_sec:   float
    roofline_tput_tok_per_sec:     float
    efficiency_ratio:              float
    vram_fits:                     bool = Field(
        description="True unless memory_fit_verdict is 'does_not_fit' — i.e. "
                     "true for both 'fits' and 'tight'. Check memory_fit_verdict "
                     "for the three-tier detail; a 'tight' GPU is expected to run "
                     "but with little headroom for allocator fragmentation."
    )
    memory_fit_verdict:            MemoryFitVerdict
    kv_cache_gb:                   float
    memory_total_gb:               float
    vram_utilization:              float
    model_size_gb:                 float
    has_training_data:             bool = Field(
        description="True unless training_data_tier is 'none'. Check "
                     "training_data_tier for whether that data actually clears "
                     "this project's 100-row-per-GPU reliability floor — "
                     "'below_floor' GPUs have real data but less than 'sufficient' ones."
    )
    training_data_tier:            TrainingDataTier
    meta:                           Optional[MetaOut] = None


class RecommendRequest(BaseModel):
    model_name:                str          = Field(..., max_length=100, examples=["llama2-70b"])
    scenario:                  Scenario     = "Offline"
    accuracy_tier:             AccuracyTier = "99"
    framework:                 Framework    = "vllm"
    batch_size:                int = Field(32,   ge=1,  le=256)
    input_tokens:              int = Field(2048, ge=64, le=8192)
    output_tokens:             int = Field(256,  ge=1,  le=4096)
    budget_per_gpu_hr:         Optional[float] = Field(None, gt=0, examples=[4.0])
    min_throughput_tok_per_sec: Optional[float] = Field(None, gt=0)
    ranking_objective:         RankingObjective = Field(
        "tokens_per_dollar",
        description="Scalar the Pareto-optimal (rank-1) set is sorted by. "
                     "'tokens_per_watt' and 'lowest_cost_per_million_tokens' need "
                     "TDP/pricing data respectively — entries missing it sort last.",
    )

    @field_validator("budget_per_gpu_hr", "min_throughput_tok_per_sec", mode="before")
    @classmethod
    def _coerce_none(cls, v: object) -> object:
        return None if v == 0 else v


class GpuResult(BaseModel):
    gpu_id:                      str
    gpu_name:                    str
    vendor:                      str
    model_name:                  str
    scenario:                    str
    accuracy_tier:                str
    framework:                   str
    pred_throughput_tok_per_sec: float
    roofline_tput_tok_per_sec:   float
    efficiency_ratio:            float
    vram_fits:                   bool = Field(
        description="True unless memory_fit_verdict is 'does_not_fit' — i.e. "
                     "true for both 'fits' and 'tight'. Check memory_fit_verdict "
                     "for the three-tier detail; a 'tight' GPU is expected to run "
                     "but with little headroom for allocator fragmentation."
    )
    memory_fit_verdict:          MemoryFitVerdict
    kv_cache_gb:                 float
    memory_total_gb:             float
    vram_utilization:            float
    model_size_gb:               float
    has_training_data:           bool = Field(
        description="True unless training_data_tier is 'none'. Check "
                     "training_data_tier for whether that data actually clears "
                     "this project's 100-row-per-GPU reliability floor — "
                     "'below_floor' GPUs have real data but less than 'sufficient' ones."
    )
    training_data_tier:          TrainingDataTier
    vram_gb:                     Optional[float]
    price_per_gpu_hr:            Optional[float]
    vram_headroom:               float
    cost_efficiency:             Optional[float]
    throughput:                  float
    watts:                       Optional[float] = Field(
        description="GPU TDP in watts (gpu_specs.yaml tdp_w), the third Pareto "
                     "axis — None only if a future SKU ships with no "
                     "recorded TDP."
    )
    tokens_per_watt:             Optional[float] = Field(
        description="pred_throughput_tok_per_sec / watts. None when watts is "
                     "unknown or throughput is 0 (a filtered/rejected entry)."
    )
    cost_per_million_tokens:     Optional[float] = Field(
        description="USD per 1M tokens served. None when price is "
                     "unknown or throughput is 0 (a filtered/rejected entry)."
    )


class FilteredGpuResult(GpuResult):
    reject_reason: str


class InfeasibilityReason(BaseModel):
    category: str = Field(
        description="One of precision_unsupported, memory_does_not_fit, "
                     "over_budget, throughput_below_minimum, other."
    )
    gpu_ids: list[str]


class Infeasibility(BaseModel):
    message: str
    reasons: list[InfeasibilityReason]
    relaxable: list[str] = Field(
        description="Human-readable suggestions for which request field(s) "
                     "to relax to admit at least one candidate."
    )


class WorkloadSummary(BaseModel):
    model_name:                str
    scenario:                  str
    accuracy_tier:             str
    framework:                 str
    model_size_gb:             float
    batch_size:                int
    input_tokens:              int
    output_tokens:             int
    budget_per_gpu_hr:         Optional[float]
    min_throughput_tok_per_sec: Optional[float]
    ranking_objective:         RankingObjective


class RecommendResponse(BaseModel):
    frontier:  list[GpuResult]
    dominated: list[GpuResult]
    filtered:  list[FilteredGpuResult]
    workload:  WorkloadSummary
    top_recommendation: Optional[GpuResult] = Field(
        description="frontier[0] (already sorted by ranking_objective), or "
                     "null iff frontier is empty — see infeasibility."
    )
    infeasibility: Optional[Infeasibility] = Field(
        description="Populated iff frontier is empty: which constraint(s) "
                     "eliminated which GPUs, and what to relax. Null whenever "
                     "top_recommendation is non-null."
    )
    meta: Optional[MetaOut] = None

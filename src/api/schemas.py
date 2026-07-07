"""Pydantic request/response schemas for the GPU Perf Prophet API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


Scenario      = Literal["Offline", "Server"]
AccuracyTier  = Literal["base", "99", "99.9"]
Framework     = Literal["vllm", "tensorrt", "rocm_other", "other"]


class PredictRequest(BaseModel):
    gpu_id:        str          = Field(..., max_length=100, examples=["mi300x"])
    model_name:    str          = Field(..., max_length=100, examples=["llama2-70b"])
    scenario:      Scenario     = "Offline"
    accuracy_tier: AccuracyTier = "99"
    framework:     Framework    = "vllm"


class PredictResponse(BaseModel):
    gpu_id:                        str
    model_name:                    str
    scenario:                      str
    accuracy_tier:                 str
    framework:                     str
    pred_throughput_tok_per_sec:   float
    roofline_tput_tok_per_sec:     float
    efficiency_ratio:              float
    vram_fits:                     bool
    model_size_gb:                 float


class RecommendRequest(BaseModel):
    model_name:                str          = Field(..., max_length=100, examples=["llama2-70b"])
    scenario:                  Scenario     = "Offline"
    accuracy_tier:             AccuracyTier = "99"
    framework:                 Framework    = "vllm"
    budget_per_gpu_hr:         Optional[float] = Field(None, gt=0, examples=[4.0])
    min_throughput_tok_per_sec: Optional[float] = Field(None, gt=0)

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
    vram_fits:                   bool
    model_size_gb:               float
    vram_gb:                     Optional[float]
    price_per_gpu_hr:            Optional[float]
    vram_headroom:               float
    cost_efficiency:             Optional[float]
    throughput:                  float


class FilteredGpuResult(GpuResult):
    reject_reason: str


class WorkloadSummary(BaseModel):
    model_name:                str
    scenario:                  str
    accuracy_tier:             str
    framework:                 str
    model_size_gb:             float
    budget_per_gpu_hr:         Optional[float]
    min_throughput_tok_per_sec: Optional[float]


class RecommendResponse(BaseModel):
    frontier:  list[GpuResult]
    dominated: list[GpuResult]
    filtered:  list[FilteredGpuResult]
    workload:  WorkloadSummary

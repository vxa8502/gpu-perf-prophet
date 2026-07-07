"""
Train and save the production XGBoost model for GPU Perf Prophet.

Reads  : data/processed/mlperf_features.parquet
Writes : data/models/prophet_v1.json
         data/models/feature_metadata.json

Run once LOGO-CV evaluation is complete:
    python -m src.models.train_final

The feature encoding mirrors notebooks/03_model_training.ipynb cell 3 exactly
so the saved model is compatible with GpuPredictor in src/models/predictor.py.

Training scope
--------------
This script trains on ALL rows with a valid target (~1112 rows), including
out-of-scope GPUs (B200, H200 NVL, etc.).  The training notebook's SHAP
analysis used only the 649 in-scope rows, but LOGO-CV training folds always
contained out-of-scope rows, so the production model mirrors what the CV folds
actually saw.  Out-of-scope GPUs are never served to users (gated by
gpu_in_model_scope in GpuRecommender); they are included here purely as
additional training signal about architectural trends.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.predictor import FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TARGET = "efficiency_ratio"
THROUGHPUT_COL = "throughput_tok_per_sec_per_gpu"

# Same hyperparameters as the training notebook's final model (no early stopping).
# n_estimators set to 250 — midpoint of observed best_iteration range across
# folds (77–341); conservative to avoid overfitting without an eval set.
PROD_PARAMS: dict = {
    "n_estimators": 250,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    "random_state": 42,
}

FEATURES_PATH = Path("data/processed/mlperf_features.parquet")
MODEL_DIR = Path("data/models")


def _encode(feat: pd.DataFrame) -> pd.DataFrame:
    """Apply the same categorical encoding as notebook cell 3."""
    df = feat.copy()

    df["scenario_offline"] = (df["scenario"] == "Offline").astype(int)

    df["fw_tensorrt"] = (df["framework_family"] == "tensorrt").astype(int)
    df["fw_vllm"] = (df["framework_family"] == "vllm").astype(int)
    df["fw_rocm_other"] = (df["framework_family"] == "rocm_other").astype(int)

    df["is_cdna4"] = (df["amd_arch_gen"] == 2).astype(int)

    return df


def train_and_save(
    features_path: Path = FEATURES_PATH,
    model_dir: Path = MODEL_DIR,
) -> xgb.XGBRegressor:
    feat = pd.read_parquet(features_path)
    log.info("Loaded %d rows from %s", len(feat), features_path)

    df = _encode(feat)

    # Drop rows where target is NaN or infinite.
    valid = df[TARGET].notna() & np.isfinite(df[TARGET])
    df = df[valid].reset_index(drop=True)
    log.info("Training rows (valid target): %d", len(df))

    # Variance gate — matches the training notebook's assertion.
    # assert compiled away under python -O; use explicit raise.
    dead = [c for c in FEATURE_COLS if df[c].nunique() <= 1]
    if dead:
        raise ValueError(f"Dead features (constant across all training rows): {dead}")

    X = df[FEATURE_COLS]
    y = df[TARGET]

    model = xgb.XGBRegressor(**PROD_PARAMS)
    model.fit(X, y)
    log.info("Model trained: %d trees, %d features", PROD_PARAMS["n_estimators"], len(FEATURE_COLS))

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "prophet_v1.json"
    meta_path = model_dir / "feature_metadata.json"

    # Write both artifacts atomically via temp-file + os.replace().
    # If this process is killed between the two writes, the reader
    # (GpuPredictor.__init__) sees either the old pair or the new pair —
    # never a mixed state that would silently mismatch feature_cols.
    tmp_model = model_path.with_name(model_path.stem + ".tmp.json")
    model.save_model(str(tmp_model))
    tmp_model.replace(model_path)
    log.info("Saved model → %s", model_path)

    meta = {
        "feature_cols": FEATURE_COLS,
        "target": TARGET,
        "model_version": "v1",
        "prod_params": PROD_PARAMS,
        "n_training_rows": len(df),
    }
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    with tmp_meta.open("w") as f:
        json.dump(meta, f, indent=2)
    tmp_meta.replace(meta_path)
    log.info("Saved metadata → %s", meta_path)

    return model


if __name__ == "__main__":
    train_and_save()

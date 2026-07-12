"""Train and save the production XGBoost model on ALL ~1112 valid-target rows (incl. out-of-scope GPUs, mirroring what LOGO-CV folds saw), using notebooks/03_model_training.ipynb cell 3's exact feature encoding for GpuPredictor compatibility."""

from __future__ import annotations

import json
import logging
import stat as _stat
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.manifest import corpus_sha256, load_manifest, sources_present, verify_manifest
from src.models.predictor import FEATURE_COLS

# Written by notebooks/03_model_training.ipynb (the only place that fits out-of-fold models); reading it here is optional/best-effort since train_and_save() trains the production model on ALL rows with no hold-out and so has no LOGO-CV metrics of its own.
LOGO_CV_METRICS_PATH = Path("data/models/logo_cv_metrics.json")
# Same cap predictor.py applies to feature_metadata.json — same trust tier (build artifact, not user input) and comparably small.
_MAX_VALIDATION_METRICS_BYTES: int = 1 * 1024 * 1024  # 1 MB

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TARGET = "efficiency_ratio"
THROUGHPUT_COL = "throughput_tok_per_sec_per_gpu"

# Same hyperparameters as the training notebook's final model (no early stopping); n_estimators=250 is the midpoint of observed best_iteration across folds (77-341), conservative to avoid overfitting without an eval set.
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


def _load_validation_metrics(path: Path = LOGO_CV_METRICS_PATH) -> dict | None:
    """Best-effort load of the LOGO-CV metrics notebooks/03 writes: returns None (not a raised error) when absent (e.g. fresh clone, notebook never re-run), but once present gets the same symlink+size guard as every other integrity-sensitive file this codebase reads, so a corrupted or maliciously-symlinked file fails loudly rather than silently reading tampered content."""
    try:
        st = path.lstat()
    except OSError:
        return None
    if _stat.S_ISLNK(st.st_mode):
        raise ValueError(f"Validation metrics path is a symlink (refused): {path}")
    if st.st_size > _MAX_VALIDATION_METRICS_BYTES:
        raise ValueError(
            f"Validation metrics file too large ({st.st_size} bytes > "
            f"{_MAX_VALIDATION_METRICS_BYTES}): {path}"
        )
    with path.open() as f:
        return json.load(f)


def train_and_save(
    features_path: Path = FEATURES_PATH,
    model_dir: Path = MODEL_DIR,
) -> xgb.XGBRegressor:
    # Verify the corpus's source snapshots still match data_manifest.lock (warn-only, see manifest.py's docstring for why); result is captured so feature_metadata.json can record verification status, with n_sources_present/total disambiguating "clean" from "mostly unchecked" since absent gitignored sources are skipped not flagged; manifest loaded once and passed to both calls to avoid re-parsing the lock file (~1.2ms/call) per call site.
    locked_manifest = load_manifest()
    manifest_mismatches = verify_manifest(locked=locked_manifest)
    n_sources_present, n_sources_total = sources_present(locked=locked_manifest)

    # Hash the exact corpus file before reading it, so feature_metadata.json records the snapshot that actually produced the model.
    corpus_hash = corpus_sha256(features_path)

    feat = pd.read_parquet(features_path)
    log.info("Loaded %d rows from %s", len(feat), features_path)

    df = _encode(feat)

    # Drop rows where target is NaN or infinite.
    valid = df[TARGET].notna() & np.isfinite(df[TARGET])
    df = df[valid].reset_index(drop=True)
    log.info("Training rows (valid target): %d", len(df))

    # Variance gate matching the training notebook's assertion; explicit raise (not assert) since assert is compiled away under python -O.
    dead = [c for c in FEATURE_COLS if df[c].nunique() <= 1]
    if dead:
        raise ValueError(f"Dead features (constant across all training rows): {dead}")

    X = df[FEATURE_COLS]
    y = df[TARGET]

    model = xgb.XGBRegressor(**PROD_PARAMS)
    model.fit(X, y)
    log.info("Model trained: %d trees, %d features", PROD_PARAMS["n_estimators"], len(FEATURE_COLS))

    validation_metrics = _load_validation_metrics()

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "prophet_v1.json"
    meta_path = model_dir / "feature_metadata.json"

    # Write both artifacts atomically via temp-file + os.replace(), so a kill mid-write leaves the reader (GpuPredictor.__init__) seeing the old pair or the new pair, never a mismatched mix.
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
        # Links this artifact to the exact corpus snapshot it trained on (lineage: prediction -> feature_metadata.json -> data_manifest.lock -> pinned MLPerf commits); manifest_verified confirms *this* corpus's pinning was actually checked clean, not just that a hash exists.
        "corpus_sha256": corpus_hash,
        "corpus_path": str(features_path),
        "manifest_verified": len(manifest_mismatches) == 0,
        # Disambiguates manifest_verified: "5/5" (every locked source present and clean) reads very differently from "1/5" (gitignored MLPerf mirrors simply absent, the normal fresh-clone/CI state), even though both give manifest_verified: true.
        "manifest_sources_present": n_sources_present,
        "manifest_sources_total": n_sources_total,
        # GPUs with zero rows here extrapolate purely from specs; served via GpuPredictor.has_training_data() so callers can disclose this instead of presenting an extrapolated number with false confidence.
        "trained_gpu_ids": sorted(df["canonical_gpu_id"].unique().tolist()),
        # Per-GPU row counts behind GpuPredictor.training_data_tier(): a boolean "has any data" can't tell a GPU with 1 row from one with 1,000 — three of eight served GPUs (the AMD SKUs) sit under the 100-row-per-GPU floor despite having nonzero data.
        "trained_gpu_row_counts": {
            str(k): int(v) for k, v in df["canonical_gpu_id"].value_counts().items()
        },
        # LOGO-CV out-of-fold metrics from notebooks/03, not this run's own result (train_and_save() fits on ALL rows with no hold-out); None if the notebook wasn't re-run since the last corpus rebuild.
        "validation_metrics": validation_metrics,
    }
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    with tmp_meta.open("w") as f:
        json.dump(meta, f, indent=2)
    tmp_meta.replace(meta_path)
    log.info("Saved metadata → %s", meta_path)

    return model


if __name__ == "__main__":
    train_and_save()

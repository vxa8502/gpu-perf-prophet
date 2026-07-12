"""
Train and save the production XGBoost model for GPU Perf Prophet.

Reads  : data/processed/mlperf_features.parquet
         data/models/logo_cv_metrics.json (optional — see _load_validation_metrics)
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
import stat as _stat
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.manifest import corpus_sha256, load_manifest, sources_present, verify_manifest
from src.models.predictor import FEATURE_COLS

# Per-fold LOGO-CV validation metrics are written by
# notebooks/03_model_training.ipynb, the only place that actually fits
# out-of-fold models — train_and_save() trains the *production* model on
# ALL rows with no hold-out, so it has no LOGO-CV metrics of its own to
# report. Reading this file is optional and best-effort: a normal
# train_and_save() run must not fail just because the notebook wasn't
# freshly re-run first.
LOGO_CV_METRICS_PATH = Path("data/models/logo_cv_metrics.json")
# Same cap predictor.py applies to feature_metadata.json — this file lives in
# the same data/models/ directory at the same trust tier (a build artifact,
# not user input) and is comparably small (a few KB of per-fold numbers).
_MAX_VALIDATION_METRICS_BYTES: int = 1 * 1024 * 1024  # 1 MB

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


def _load_validation_metrics(path: Path = LOGO_CV_METRICS_PATH) -> dict | None:
    """Best-effort load of the LOGO-CV metrics notebooks/03 writes.

    Returns None (not a raised error) when absent — e.g. a fresh clone or a
    train-only CI run that never re-executed the notebook — so this is
    informational, matching the warn-only-not-hard-refuse convention
    src/data/manifest.py already established for optional provenance data.
    "Optional" only covers absence, though: once the file is present, it gets
    the same symlink+size guard as every other integrity-sensitive file this
    codebase reads (gpu_spec_db.load_specs, recommender._load_pricing,
    predictor.py's model-artifact loader, manifest.py's file readers) — a
    corrupted or maliciously-symlinked metrics file fails loudly rather than
    silently reading tampered or unbounded content into a committed artifact.
    """
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
    # Verify the training corpus's source snapshots (MLPerf
    # submission mirrors + AMD Dev Cloud calibration CSV) still match
    # data/data_manifest.lock. Warn-only, not hard-refuse — see
    # src/data/manifest.py's module docstring for why. The result is
    # captured (not discarded) so feature_metadata.json can record whether
    # this build was verified clean — without it, corpus_sha256 alone only
    # proves *which* corpus trained the model, not whether that corpus's
    # sources were confirmed to match data_manifest.lock at the time. That
    # warning would otherwise only ever exist in this run's console output.
    # manifest_verified alone can't be trusted at face value either: a locked
    # source that's simply absent (the common case for the gitignored MLPerf
    # mirrors on a fresh clone or in CI) is skipped, not flagged, by
    # verify_manifest()'s own "absence isn't drift" design — so a "clean"
    # result could mean every source matched, or that most were never
    # checked at all. n_sources_present/n_sources_total give a reader that
    # missing denominator.
    #
    # Loaded once and passed to both calls — verify_manifest() and
    # sources_present() each independently call load_manifest() (a symlink
    # guard + file read + YAML parse of data_manifest.lock) when not given
    # a pre-loaded manifest, which would otherwise mean reading and parsing
    # the same lock file twice for the same call site (measured ~1.2ms/call,
    # small in absolute terms but the same "recompute once, not per call"
    # class of fix this codebase has made repeatedly regardless of
    # magnitude.
    locked_manifest = load_manifest()
    manifest_mismatches = verify_manifest(locked=locked_manifest)
    n_sources_present, n_sources_total = sources_present(locked=locked_manifest)

    # Hash the exact corpus file this run trains on, before reading
    # it, so feature_metadata.json records the snapshot that actually
    # produced the model — not just "a" corpus that happened to be nearby.
    corpus_hash = corpus_sha256(features_path)

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

    validation_metrics = _load_validation_metrics()

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
        # Links this artifact back to the exact corpus snapshot it was
        # trained on — the link between the built corpus and this model in
        # the lineage chain served prediction → feature_metadata.json →
        # data_manifest.lock → pinned MLPerf commits (data_manifest.lock
        # pins the raw sources that built this corpus). manifest_verified
        # records whether that pinning was actually confirmed clean for
        # *this* corpus, not just that a hash exists — a corpus_sha256
        # alone can't distinguish a build made after `python -m
        # src.data.manifest` refreshed the lock from one made against a
        # since-drifted lock nobody re-pinned.
        "corpus_sha256": corpus_hash,
        "corpus_path": str(features_path),
        "manifest_verified": len(manifest_mismatches) == 0,
        # Disambiguates manifest_verified: e.g. "5/5" (every locked source
        # present and confirmed clean) reads very differently from "1/5"
        # (only the git-tracked calibration CSV was actually checked — the
        # gitignored MLPerf mirrors were simply absent, the normal state on
        # a fresh clone or in CI), even though both produce
        # manifest_verified: true.
        "manifest_sources_present": n_sources_present,
        "manifest_sources_total": n_sources_total,
        # GPUs with zero rows here never had a single real measurement behind
        # their predictions — the model extrapolates purely from specs. Served
        # via GpuPredictor.has_training_data() so callers can disclose this
        # rather than presenting an extrapolated number with false confidence.
        "trained_gpu_ids": sorted(df["canonical_gpu_id"].unique().tolist()),
        # Per-GPU row counts behind GpuPredictor.training_data_tier(): a
        # boolean "has any data" (above) can't tell a GPU with 1 row apart
        # from one with 1,000 — three of the eight served GPUs (the AMD SKUs,
        # per data/data_card.md) sit under this project's 100-row-per-GPU
        # Must-have floor despite having nonzero data, and the boolean alone
        # reported them identically to GPUs with 2-3x more rows.
        "trained_gpu_row_counts": {
            str(k): int(v) for k, v in df["canonical_gpu_id"].value_counts().items()
        },
        # The sidecar should carry "feature schema, HPs,
        # per-fold metrics, corpus SHA" — this project had the first, third,
        # and fourth for a while but never the metrics themselves,
        # leaving no way to answer "how good was the model this corpus
        # produced" from the committed artifact alone. These are LOGO-CV out-of-fold metrics from notebooks/03 —
        # NOT this run's own training result, since train_and_save() fits on
        # ALL rows with no hold-out and so has no held-out score to report.
        # None if the notebook wasn't re-run since the last corpus rebuild.
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

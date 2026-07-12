"""
Tests for src/models/train_final.py's `_load_validation_metrics`.

Strategy: unit-test the pure load function against a synthetic tmp_path
file, not the real data/models/logo_cv_metrics.json — these tests must not
depend on notebooks/03 having been re-run recently.
"""

from __future__ import annotations

import json

import pytest

from src.models import train_final
from src.models.train_final import _load_validation_metrics


class TestLoadValidationMetrics:
    def test_returns_none_when_file_absent(self, tmp_path):
        missing = tmp_path / "logo_cv_metrics.json"
        assert _load_validation_metrics(missing) is None

    def test_returns_parsed_dict_when_present(self, tmp_path):
        path = tmp_path / "logo_cv_metrics.json"
        payload = {"primary": {"mape": 0.234, "smape": 0.212, "n_folds": 5}}
        path.write_text(json.dumps(payload))
        assert _load_validation_metrics(path) == payload

    def test_default_path_does_not_raise(self):
        # Whether or not the real file happens to exist on this machine,
        # calling with no argument must never raise — it's a best-effort
        # informational read, not a hard requirement of train_and_save().
        result = _load_validation_metrics()
        assert result is None or isinstance(result, dict)

    def test_refuses_symlinked_file(self, tmp_path):
        """Same guard predictor.py applies to feature_metadata.json/
        prophet_v1.json — a symlinked metrics file could point at
        attacker-controlled content that gets committed into
        feature_metadata.json's validation_metrics field with no warning."""
        real = tmp_path / "real.json"
        real.write_text(json.dumps({"primary": {"mape": 0.0}}))
        link = tmp_path / "link.json"
        link.symlink_to(real)
        with pytest.raises(ValueError, match="symlink"):
            _load_validation_metrics(link)

    def test_refuses_oversized_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(train_final, "_MAX_VALIDATION_METRICS_BYTES", 10)
        p = tmp_path / "logo_cv_metrics.json"
        p.write_text(json.dumps({"primary": {"mape": 0.234}}))
        with pytest.raises(ValueError, match="too large"):
            _load_validation_metrics(p)

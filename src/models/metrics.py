"""Evaluation metrics co-reported alongside MAPE: sMAPE and median APE, as plain functions (not sklearn wrappers) so LOGO-CV notebooks and future eval code share one definition."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_EPS = np.finfo(float).eps


def smape(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Symmetric MAPE: mean(2*|pred-true| / (|true|+|pred|)), as a fraction in [0, 2]; denominator floored at machine epsilon (matches sklearn's convention) so true==pred==0 gives 0 instead of divide-by-zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, _EPS, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def median_ape(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Median absolute percentage error, as a fraction; robust to the single-worst-row blowups that can dominate a small-N fold's mean MAPE (same epsilon-floored denominator convention as smape())."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true)
    denom = np.where(denom == 0, _EPS, denom)
    return float(np.median(np.abs(y_pred - y_true) / denom))

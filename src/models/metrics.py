"""
Evaluation metrics co-reported alongside MAPE.

MAPE is asymmetric and explodes for near-zero actuals.
sMAPE bounds the penalty for over- vs under-prediction symmetrically; median
APE is robust to the single-worst-row blowups that can dominate a small-N
fold's mean. Both are plain functions (not sklearn wrappers) so LOGO-CV
notebooks and any future evaluation code share one definition rather than
each hand-rolling the formula.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_EPS = np.finfo(float).eps


def smape(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Symmetric MAPE: mean(2*|pred-true| / (|true|+|pred|)), as a fraction in [0, 2].

    Denominator is floored at machine epsilon (matches sklearn's
    mean_absolute_percentage_error convention) so a true==pred==0 row
    contributes 0 rather than raising a divide-by-zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, _EPS, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def median_ape(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Median absolute percentage error, as a fraction (median analogue of MAPE).

    Robust to the single-worst-row blowups that can dominate a small-N
    fold's *mean* MAPE. Denominator floored at machine epsilon, same
    convention as smape() above.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true)
    denom = np.where(denom == 0, _EPS, denom)
    return float(np.median(np.abs(y_pred - y_true) / denom))

"""
Tests for src/models/metrics.py (sMAPE, median APE).

Strategy: hand-computed expected values (no reliance on sklearn or the
functions' own internals as the "expected" source), plus edge cases for the
zero-denominator floor.
"""

from __future__ import annotations

import math
import sys

from src.models.metrics import median_ape, smape


class TestSmape:
    def test_single_row_hand_computed(self):
        # 2*|150-100| / (100+150) = 100/250 = 0.4
        assert math.isclose(smape([100], [150]), 0.4)

    def test_multi_row_hand_computed(self):
        # row1: 2*10/210 = 0.0952380952...
        # row2: 2*20/380 = 0.1052631579...
        # mean = 0.1002506265...
        expected = (2 * 10 / 210 + 2 * 20 / 380) / 2
        assert math.isclose(smape([100, 200], [110, 180]), expected)

    def test_perfect_prediction_is_zero(self):
        assert smape([1, 2, 3], [1, 2, 3]) == 0.0

    def test_symmetric_in_over_and_under_prediction(self):
        # sMAPE's whole point vs. plain MAPE: overshoot and undershoot by the
        # same absolute amount around the same true value score identically.
        over = smape([100], [150])
        under = smape([150], [100])
        assert math.isclose(over, under)

    def test_zero_true_and_zero_pred_does_not_raise_and_is_zero(self):
        # Denominator floored at machine epsilon; numerator is also 0, so the
        # ratio is 0, not NaN/inf.
        assert smape([0], [0]) == 0.0

    def test_bounded_in_zero_to_two(self):
        # sMAPE's fractional form is bounded in [0, 2] by construction
        # (numerator <= denominator always, times 2).
        result = smape([1, 500, 0.001], [1000, 1, 5])
        assert 0.0 <= result <= 2.0


class TestMedianApe:
    def test_hand_computed_odd_count(self):
        # ape: 0.10, 0.10, 0.60 -> median 0.10
        assert math.isclose(median_ape([100, 200, 50], [110, 180, 80]), 0.10)

    def test_robust_to_single_outlier_unlike_mean(self):
        # One row is wildly wrong; median APE should ignore it, mean would not.
        y_true = [100, 100, 100, 100, 100]
        y_pred = [100, 100, 100, 100, 100000]  # one row off by 10,000%
        assert math.isclose(median_ape(y_true, y_pred), 0.0)

    def test_perfect_prediction_is_zero(self):
        assert median_ape([1, 2, 3], [1, 2, 3]) == 0.0

    def test_zero_true_denominator_floored_not_raising(self):
        # Should not raise ZeroDivisionError/produce NaN; result is a large
        # finite number (division by *machine epsilon specifically*, not
        # some other arbitrary floor). Pinning the exact expected value
        # (not just "finite and positive") matters: a much looser floor
        # (e.g. 1.0 instead of epsilon) would still be finite and positive
        # but silently wrong by ~16 orders of magnitude — "finite and > 0"
        # alone doesn't distinguish a correct epsilon floor from a broken
        # one (confirmed via mutation: swapping the floor to 1.0 left this
        # assertion, in its old finite-and-positive form, still passing).
        # sys.float_info.epsilon, not metrics._EPS, so a typo'd _EPS binding
        # inside the module under test can't also poison this expectation.
        expected = 5.0 / sys.float_info.epsilon
        result = median_ape([0], [5])
        assert math.isclose(result, expected)

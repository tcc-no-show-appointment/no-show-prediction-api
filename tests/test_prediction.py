import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app.services.prediction_service import _normalize_probability
from app.models.schemas import PredictionResponse


# ---------------------------------------------------------------------------
# _normalize_probability
# ---------------------------------------------------------------------------

class TestNormalizeProbability:
    def test_threshold_maps_to_half(self):
        """The threshold value must always normalize to exactly 0.5."""
        for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            assert _normalize_probability(t, t) == pytest.approx(0.5)

    def test_zero_maps_to_zero(self):
        for t in [0.3, 0.5, 0.7]:
            assert _normalize_probability(0.0, t) == pytest.approx(0.0)

    def test_one_maps_to_one(self):
        for t in [0.3, 0.5, 0.7]:
            assert _normalize_probability(1.0, t) == pytest.approx(1.0)

    def test_below_threshold_linear(self):
        """Values below threshold map linearly into [0, 0.5]."""
        result = _normalize_probability(0.15, 0.3)
        assert result == pytest.approx(0.15 / (2 * 0.3))

    def test_above_threshold_linear(self):
        """Values above threshold map linearly into [0.5, 1]."""
        result = _normalize_probability(0.65, 0.3)
        assert result == pytest.approx(0.5 + (0.65 - 0.3) / (2 * 0.7))

    def test_output_in_range(self):
        """Normalized output must always be in [0, 1]."""
        threshold = 0.4
        for p in [i / 10 for i in range(11)]:
            result = _normalize_probability(p, threshold)
            assert 0.0 <= result <= 1.0

    def test_invalid_threshold_passthrough(self):
        """Threshold of 0 or 1 should return p unchanged."""
        assert _normalize_probability(0.6, 0.0) == pytest.approx(0.6)
        assert _normalize_probability(0.6, 1.0) == pytest.approx(0.6)

    def test_preserves_order_below_threshold(self):
        """Normalization must be monotone below threshold."""
        t = 0.35
        values = [0.0, 0.1, 0.2, 0.3, 0.35]
        normalized = [_normalize_probability(v, t) for v in values]
        assert normalized == sorted(normalized)

    def test_preserves_order_above_threshold(self):
        """Normalization must be monotone above threshold."""
        t = 0.35
        values = [0.35, 0.5, 0.7, 0.9, 1.0]
        normalized = [_normalize_probability(v, t) for v in values]
        assert normalized == sorted(normalized)


# ---------------------------------------------------------------------------
# PredictionResponse schema
# ---------------------------------------------------------------------------

class TestPredictionResponseSchema:
    def _valid_payload(self):
        return {
            "prediction": 0,
            "prediction_label": "show",
            "probability_show": 0.65,
            "probability_no_show": 0.35,
            "probability_no_show_normalized": 0.45,
            "threshold": 0.40,
        }

    def test_valid_response(self):
        resp = PredictionResponse(**self._valid_payload())
        assert resp.prediction == 0
        assert resp.prediction_label == "show"
        assert resp.threshold == pytest.approx(0.40)

    def test_normalized_field_present(self):
        resp = PredictionResponse(**self._valid_payload())
        assert hasattr(resp, "probability_no_show_normalized")
        assert hasattr(resp, "threshold")

    def test_no_show_prediction(self):
        payload = self._valid_payload()
        payload["prediction"] = 1
        payload["prediction_label"] = "no-show"
        resp = PredictionResponse(**payload)
        assert resp.prediction == 1

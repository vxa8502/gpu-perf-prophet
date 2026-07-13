"""Unit tests for app/api_client.py — the HTTP client the Streamlit UI uses to call FastAPI."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import requests

from app.api_client import ApiError, ApiUnavailableError, recommend


def _response(status_code: int, json_body: object = None, text: str = "") -> Mock:
    resp = Mock()
    resp.status_code = status_code
    resp.text = text
    if json_body is None:
        resp.json.side_effect = ValueError("no JSON body")
    else:
        resp.json.return_value = json_body
    return resp


class TestRecommendSuccess:
    def test_returns_parsed_json_on_200(self):
        with patch("app.api_client.requests.post", return_value=_response(200, {"frontier": []})):
            result = recommend(model_name="llama2-70b")
        assert result == {"frontier": []}

    def test_posts_payload_as_json_to_recommend_route(self):
        mock_post = Mock(return_value=_response(200, {}))
        with patch("app.api_client.requests.post", mock_post):
            recommend(model_name="llama2-70b", ranking_objective="tokens_per_watt")
        _, kwargs = mock_post.call_args
        assert mock_post.call_args[0][0].endswith("/recommend")
        assert kwargs["json"] == {"model_name": "llama2-70b", "ranking_objective": "tokens_per_watt"}


class TestRecommendErrorResponse:
    def test_422_raises_api_error_with_detail(self):
        with patch("app.api_client.requests.post", return_value=_response(422, {"detail": "bad input"})):
            with pytest.raises(ApiError) as exc_info:
                recommend(model_name="not-a-model")
        assert exc_info.value.status_code == 422
        assert exc_info.value.detail == "bad input"

    def test_429_plain_text_body_falls_back_to_response_text(self):
        # The rate-limiter's 429 response (src/api/main.py) is plain text, not JSON.
        with patch(
            "app.api_client.requests.post",
            return_value=_response(429, json_body=None, text="Rate limit exceeded"),
        ):
            with pytest.raises(ApiError) as exc_info:
                recommend(model_name="llama2-70b")
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail == "Rate limit exceeded"


class TestRecommendUnavailable:
    def test_connection_error_retries_then_raises_api_unavailable(self):
        mock_post = Mock(side_effect=requests.exceptions.ConnectionError("refused"))
        with patch("app.api_client.requests.post", mock_post), patch("app.api_client.time.sleep"):
            with pytest.raises(ApiUnavailableError):
                recommend(model_name="llama2-70b")
        assert mock_post.call_count == 10  # _CONNECT_RETRIES

    def test_recovers_if_api_becomes_reachable_before_retries_exhausted(self):
        mock_post = Mock(
            side_effect=[
                requests.exceptions.ConnectionError("refused"),
                requests.exceptions.ConnectionError("refused"),
                _response(200, {"frontier": []}),
            ]
        )
        with patch("app.api_client.requests.post", mock_post), patch("app.api_client.time.sleep"):
            result = recommend(model_name="llama2-70b")
        assert result == {"frontier": []}
        assert mock_post.call_count == 3

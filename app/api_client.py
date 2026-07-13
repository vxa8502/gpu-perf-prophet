"""Thin HTTP client the Streamlit UI uses to call the FastAPI service running alongside it in the same container (see docker/supervisord.conf), instead of importing GpuPredictor/GpuRecommender and calling them in-process."""

from __future__ import annotations

import os
import time

import requests

API_BASE_URL = os.environ.get("GPP_API_BASE_URL", "http://127.0.0.1:8000")

# The uvicorn sibling process (docker/supervisord.conf) can still be starting when the first user interaction lands; retry connection errors only, never error *responses*, for up to ~5s.
_CONNECT_RETRIES = 10
_CONNECT_RETRY_DELAY_S = 0.5
_TIMEOUT_S = 15.0


class ApiError(Exception):
    """A well-formed error response from the API (4xx/5xx with a JSON `detail`)."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")


class ApiUnavailableError(Exception):
    """The API never became reachable (e.g. uvicorn is still starting, or crashed)."""


def recommend(**payload: object) -> dict:
    """POST /recommend and return the parsed JSON body."""
    url = f"{API_BASE_URL}/recommend"
    resp = None
    last_exc: Exception | None = None
    for _ in range(_CONNECT_RETRIES):
        try:
            resp = requests.post(url, json=payload, timeout=_TIMEOUT_S)
            break
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            time.sleep(_CONNECT_RETRY_DELAY_S)
    if resp is None:
        raise ApiUnavailableError(
            f"Could not reach the prediction API at {API_BASE_URL} after "
            f"{_CONNECT_RETRIES} attempts."
        ) from last_exc

    if resp.status_code == 200:
        return resp.json()

    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    raise ApiError(resp.status_code, str(detail))

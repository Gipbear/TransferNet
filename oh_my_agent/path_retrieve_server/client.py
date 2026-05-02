"""Client for the cached path retrieve server."""

from __future__ import annotations

from typing import Optional

import requests

from .schema import RetrieveResponse as PathRetrieveResponse


class PathRetrieveClient:
    def __init__(self, base_url: str = "http://localhost:8789", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, endpoint: str, payload: dict) -> PathRetrieveResponse:
        resp = requests.post(f"{self.base_url}{endpoint}", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return PathRetrieveResponse(**resp.json())

    def retrieve(
        self,
        question: Optional[str] = None,
        *,
        sample_index: Optional[int] = None,
        topic_entities: Optional[list[str]] = None,
        method: str = "tail_blend",
        alpha_final: float = 1.0,
        threshold: float = 0.01,
        beam_size: int = 50,
        lambda_val: float = 0.5,
    ) -> PathRetrieveResponse:
        return self._post(
            "/retrieve",
            {
                "question": question,
                "sample_index": sample_index,
                "topic_entities": topic_entities,
                "method": method,
                "alpha_final": alpha_final,
                "threshold": threshold,
                "beam_size": beam_size,
                "lambda_val": lambda_val,
            },
        )

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def info(self) -> dict:
        resp = requests.get(f"{self.base_url}/info", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

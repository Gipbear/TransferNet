"""缓存路径检索服务的 HTTP client(迁自 oh_my_agent/path_retrieve_server/client.py)。"""

from __future__ import annotations

from typing import Optional

import requests

from kgqa.retrieve.api.schema import RetrieveResponse as PathRetrieveResponse


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
        eta: float = 1.0,
        alpha_final: float | None = None,
        threshold: float = 0.01,
        beam_size: int = 50,
        lambda_val: float = 0.2,
    ) -> PathRetrieveResponse:
        if alpha_final is not None:
            eta = alpha_final
        return self._post(
            "/retrieve",
            {
                "question": question,
                "sample_index": sample_index,
                "topic_entities": topic_entities,
                "eta": eta,
                "alpha_final": eta,
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

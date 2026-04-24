"""
TransferNet MMR 路径检索服务器客户端封装。

Usage:
    from oh_my_agent.path_server.client import PathRetrievalClient

    client = PathRetrievalClient("http://localhost:8787")
    resp = client.retrieve(
        "who was vice president after kennedy died",
        topic_entities=["m.0d3k14"],
        beam_size=20,
        lambda_val=0.2,
    )
    print(resp.mmr_reason_paths)
"""

from __future__ import annotations

from typing import Optional

import requests

from .schema import RetrieveResponse as PathRetrievalResponse


class PathRetrievalClient:
    def __init__(self, base_url: str = "http://localhost:8787", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, endpoint: str, payload: dict) -> PathRetrievalResponse:
        url = f"{self.base_url}{endpoint}"
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return PathRetrievalResponse(**resp.json())

    def retrieve(
        self,
        question: str,
        *,
        topic_entities: list[str],
        hop: Optional[int] = None,
        beam_size: int = 20,
        lambda_val: float = 0.2,
        prediction_threshold: float = 0.9,
    ) -> PathRetrievalResponse:
        return self._post(
            "/retrieve",
            {
                "question": question,
                "topic_entities": topic_entities,
                "hop": hop,
                "beam_size": beam_size,
                "lambda_val": lambda_val,
                "prediction_threshold": prediction_threshold,
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

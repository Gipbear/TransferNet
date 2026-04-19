"""
本地 LLM 服务器客户端封装。

Usage:
    from oh_my_agent.llm_server.client import LLMClient

    client = LLMClient("http://localhost:8788")

    # 带 adapter 推理
    resp = client.generate("What is the capital of France?", use_adapter=True)
    print(resp.text)

    # base 模型推理（无 adapter）
    resp = client.generate("What is the capital of France?", use_adapter=False)
    print(resp.text)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class GenerateResponse:
    text: str
    used_adapter: bool
    tokens_generated: int
    elapsed_ms: float


class LLMClient:
    def __init__(self, base_url: str = "http://localhost:8788", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, endpoint: str, payload: dict) -> GenerateResponse:
        url = f"{self.base_url}{endpoint}"
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return GenerateResponse(**data)

    def generate(
        self,
        prompt: str,
        *,
        use_adapter: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> GenerateResponse:
        return self._post("/generate", {
            "prompt": prompt,
            "use_adapter": use_adapter,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "system_prompt": system_prompt,
        })

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def info(self) -> dict:
        resp = requests.get(f"{self.base_url}/info", timeout=10)
        resp.raise_for_status()
        return resp.json()

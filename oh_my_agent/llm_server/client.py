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
import time


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


class OpenAICompatibleLLMClient:
    """Client for vLLM/SGLang style OpenAI-compatible chat endpoints."""

    def __init__(
        self,
        base_url: str = "http://localhost:8788/v1",
        *,
        model: str,
        timeout: int = 120,
        api_key: str = "EMPTY",
        adapter_model: str | None = None,
    ):
        if not model:
            raise ValueError("model is required for OpenAI-compatible LLM client")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.adapter_model = adapter_model

    def generate(
        self,
        prompt: str,
        *,
        use_adapter: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> GenerateResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        model = self.adapter_model if use_adapter and self.adapter_model else self.model
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        text = choice.get("message", {}).get("content") or choice.get("text", "")
        usage = data.get("usage") or {}
        tokens_generated = int(usage.get("completion_tokens") or 0)
        return GenerateResponse(
            text=text,
            used_adapter=bool(use_adapter and self.adapter_model),
            tokens_generated=tokens_generated,
            elapsed_ms=round(elapsed_ms, 1),
        )

    def health(self) -> dict:
        resp = requests.get(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        return {"status": "ok", "models": resp.json().get("data", [])}

    def info(self) -> dict:
        return self.health()

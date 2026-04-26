"""FastAPI 应用工厂：lifespan + /generate /health /info 端点。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .client import GenerateResponse as _ClientGenerateResponse
from .engine import ModelEngine
from .scheduler import BatchScheduler


class GenerateRequest(BaseModel):
    prompt: str
    use_adapter: bool = True
    max_new_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    system_prompt: Optional[str] = None


class GenerateResponse(BaseModel):
    text: str
    used_adapter: bool
    tokens_generated: int
    elapsed_ms: float


def create_app(engine: ModelEngine, scheduler: BatchScheduler) -> FastAPI:
    """创建 FastAPI 应用实例，注入 engine 和 scheduler。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        scheduler.start()
        try:
            yield
        finally:
            scheduler.stop()

    app = FastAPI(title="LLM Local Server", version="2.0", lifespan=lifespan)

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        future = scheduler.submit(
            req.prompt, req.use_adapter, req.max_new_tokens,
            req.temperature, req.system_prompt,
        )
        try:
            result: _ClientGenerateResponse = await asyncio.wrap_future(future)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return GenerateResponse(
            text=result.text,
            used_adapter=result.used_adapter,
            tokens_generated=result.tokens_generated,
            elapsed_ms=result.elapsed_ms,
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": engine._model is not None}

    @app.get("/info")
    def info():
        return engine.info()

    return app

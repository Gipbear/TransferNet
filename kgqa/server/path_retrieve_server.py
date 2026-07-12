"""常驻检索服务（薄壳，持有一个后端）。"""
from __future__ import annotations

import argparse
from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_index: int
    beam_size: int = 50
    lambda_val: float = 0.2
    threshold: float = 0.01
    alpha_final: float = 1.0


def create_app(backend) -> FastAPI:
    app = FastAPI(title="kgqa path retrieve")

    @app.get("/health")
    def health():
        return {"status": "ok", "n": len(backend.bundle.samples)}

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest):
        result = backend.retrieve(
            req.sample_index, beam_size=req.beam_size,
            lambda_val=req.lambda_val, threshold=req.threshold, alpha_final=req.alpha_final,
        )
        return asdict(result)

    return app


def main(argv=None):
    p = argparse.ArgumentParser(description="kgqa 常驻检索服务")
    p.add_argument("--dataset", required=True)
    p.add_argument("--backend", choices=["offline"], default="offline")
    p.add_argument("--cache", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8789)
    args = p.parse_args(argv)

    import uvicorn
    from kgqa.datasets.registry import get_adapter
    from kgqa.retrieve.backends.offline import OfflineBackend
    adapter = get_adapter(args.dataset, input_dir=args.input_dir)
    backend = OfflineBackend(adapter, cache_path=args.cache)
    uvicorn.run(create_app(backend), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

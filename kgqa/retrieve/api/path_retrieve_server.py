"""常驻检索服务。

- ``create_app(backend)``:薄壳,按 sample_index 查离线后端(stage1 原有)。
- ``create_service_app(service)``:全功能服务(stage3 上移自 legacy
  oh_my_agent/path_retrieve_server/server.py):question/topic_entities 定位、
  θ 阈值 prediction、group_tails,HTTP schema 与 legacy 兼容;
  ``PATH_DROP_LOOPBACK=0`` 可关闭检索层 loopback 剔除(同环境 A/B 消融)。
  ``main`` 以全功能服务启动,供 ``scripts/path_retrieve_server.sh`` 使用。
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from . import schema


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_index: int
    beam_size: int = 50
    lambda_val: float = 0.2
    threshold: float = 0.01
    eta: float = 1.0


def create_app(backend) -> FastAPI:
    app = FastAPI(title="kgqa path retrieve")

    @app.get("/health")
    def health():
        return {"status": "ok", "n": len(backend.bundle.samples)}

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest):
        result = backend.retrieve(
            req.sample_index, beam_size=req.beam_size,
            lambda_val=req.lambda_val, threshold=req.threshold, eta=req.eta,
        )
        return asdict(result)

    return app


def create_service_app(service, *, drop_loopback: bool | None = None) -> FastAPI:
    if drop_loopback is None:
        drop_loopback = os.environ.get("PATH_DROP_LOOPBACK", "1") != "0"
    app = FastAPI(title="Cached TransferNet Path Retrieve Server", version="2.0")

    @app.post("/retrieve", response_model=schema.RetrieveResponse)
    def retrieve(req: schema.RetrieveRequest):
        try:
            result = service.retrieve(
                question=req.question,
                sample_index=req.sample_index,
                topic_entities=req.topic_entities,
                eta=req.eta,
                threshold=req.threshold,
                beam_size=req.beam_size,
                lambda_val=req.lambda_val,
                drop_loopback=drop_loopback,
            )
            return result.to_dict()
        except (KeyError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health")
    def health():
        return {"status": "ok", "cache_loaded": True}

    @app.get("/info")
    def info():
        return {"cache_loaded": True, **service.info()}

    return app


def main(argv=None):
    p = argparse.ArgumentParser(description="kgqa 常驻检索服务")
    p.add_argument("--dataset", required=True)
    p.add_argument("--backend", choices=["offline"], default="offline")
    p.add_argument("--cache", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8789)
    p.add_argument("--prediction_threshold", type=float, default=None,
                   help="e_score 预测阈值 θ(默认 0.9;group_tails/prediction/expansion 共用)")
    args = p.parse_args(argv)

    import uvicorn
    from kgqa.retrieve.datasets.registry import get_adapter
    from kgqa.retrieve.api.service import DEFAULT_PREDICTION_SCORE_THRESHOLD, PathRetrieveService
    adapter = get_adapter(args.dataset, input_dir=args.input_dir)
    threshold = (DEFAULT_PREDICTION_SCORE_THRESHOLD
                 if args.prediction_threshold is None else args.prediction_threshold)
    service = PathRetrieveService(adapter, cache_path=args.cache,
                                  prediction_threshold=threshold)
    uvicorn.run(create_service_app(service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

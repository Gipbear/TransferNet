"""HTTP server for cached offline path retrieval."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException

from .schema import RetrieveRequest, RetrieveResponse
from .service import CachedPathRetriever

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_retriever: Optional[CachedPathRetriever] = None
# 默认在检索层剔除"绕回 topic"的无效路径;设 PATH_DROP_LOOPBACK=0 可关闭(同环境 A/B 消融)
_DROP_LOOPBACK: bool = os.environ.get("PATH_DROP_LOOPBACK", "1") != "0"

app = FastAPI(title="Cached TransferNet Path Retrieve Server", version="1.0")


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    if _retriever is None:
        raise HTTPException(status_code=503, detail="score cache 未加载")
    try:
        result = _retriever.retrieve(
            question=req.question,
            sample_index=req.sample_index,
            topic_entities=req.topic_entities,
            alpha_final=req.alpha_final,
            threshold=req.threshold,
            beam_size=req.beam_size,
            lambda_val=req.lambda_val,
            drop_loopback=_DROP_LOOPBACK,
        )
        return result.to_dict()
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok", "cache_loaded": _retriever is not None}


@app.get("/info")
def info():
    if _retriever is None:
        return {"cache_loaded": False}
    return {"cache_loaded": True, **_retriever.info()}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="本地 cached TransferNet 路径检索 HTTP 服务器")
    p.add_argument("--cache", required=True, help="WebQSP dump_scores 生成的 score cache")
    p.add_argument("--input_dir", required=True, help="WebQSP 数据目录")
    p.add_argument("--port", type=int, default=8789)
    p.add_argument("--host", default="0.0.0.0")
    return p.parse_args()


def _load_retriever(args: argparse.Namespace) -> None:
    global _retriever
    log.info("加载 cached path retriever: cache=%s input_dir=%s", args.cache, args.input_dir)
    _retriever = CachedPathRetriever(cache_path=args.cache, input_dir=args.input_dir)
    log.info("cached path retriever 加载完成: %s", _retriever.info())


if __name__ == "__main__":
    args = _parse_args()
    try:
        _load_retriever(args)
    except Exception as exc:
        sys.exit(f"[Error] failed to load cached path retriever: {exc}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

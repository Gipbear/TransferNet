"""
本地 TransferNet MMR 路径检索 HTTP 服务器。

Usage:
    python -m oh_my_agent.path_server.server \
        --dataset webqsp \
        --input_dir data/input/WebQSP \
        --ckpt data/ckpt/WebQSP/model-29-0.6411.pt \
        --port 8787

API:
    POST /retrieve
        body: {
          "question": "...",
          "topic_entities": ["m.0d3k14"],
          "hop": null,
          "beam_size": 20,
          "lambda_val": 0.2,
          "prediction_threshold": 0.9
        }
        response: {
          "question": "...",
          "topics": ["m.0d3k14"],
          "hop": 2,
          "mmr_reason_paths": [{"path": [[s, r, o], ...], "log_score": -1.23}],
          "prediction": {"m.xxx": 0.953},
          "elapsed_ms": 42.0
        }

    GET /health
    GET /info
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException

from .schema import RetrieveRequest, RetrieveResponse
from .service import TransferNetPathRetriever

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_retriever: Optional[TransferNetPathRetriever] = None

app = FastAPI(title="TransferNet MMR Path Server", version="1.0")
@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    if _retriever is None:
        raise HTTPException(status_code=503, detail="TransferNet 模型未加载")
    try:
        return _retriever.retrieve(
            question=req.question,
            topic_entities=req.topic_entities,
            hop=req.hop,
            beam_size=req.beam_size,
            lambda_val=req.lambda_val,
            prediction_threshold=req.prediction_threshold,
        ).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _retriever is not None}


@app.get("/info")
def info():
    if _retriever is None:
        return {"model_loaded": False}
    return {"model_loaded": True, **_retriever.info()}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="本地 TransferNet MMR 路径检索 HTTP 服务器")
    p.add_argument("--dataset", default="webqsp", choices=["webqsp", "metaqa"])
    p.add_argument("--input_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--bert_name", default="bert-base-uncased",
                   help="WebQSP BERT/RoBERTa encoder name")
    p.add_argument("--num_steps", type=int, default=3,
                   help="MetaQA model num_steps")
    p.add_argument("--dim_word", type=int, default=300,
                   help="MetaQA word embedding dim")
    p.add_argument("--dim_hidden", type=int, default=1024,
                   help="MetaQA hidden dim")
    p.add_argument("--aux_hop", type=int, default=1, choices=[0, 1],
                   help="MetaQA aux_hop setting used at training time")
    p.add_argument("--entity_label", default=None,
                   help="可选实体名映射文件，JSON 或 TSV")
    p.add_argument("--device", default=None,
                   help="默认自动选择 cuda/cpu")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--host", default="0.0.0.0")
    return p.parse_args()


def _load_retriever(args: argparse.Namespace) -> None:
    global _retriever
    log.info("加载 TransferNet path retriever: dataset=%s ckpt=%s", args.dataset, args.ckpt)
    _retriever = TransferNetPathRetriever(
        dataset=args.dataset,
        input_dir=args.input_dir,
        ckpt=args.ckpt,
        device=args.device,
        bert_name=args.bert_name,
        num_steps=args.num_steps,
        dim_word=args.dim_word,
        dim_hidden=args.dim_hidden,
        aux_hop=args.aux_hop,
        entity_label=args.entity_label,
    )
    log.info("TransferNet path retriever 加载完成: %s", _retriever.info())


if __name__ == "__main__":
    args = _parse_args()
    try:
        _load_retriever(args)
    except Exception as exc:
        sys.exit(f"[Error] failed to load TransferNet path retriever: {exc}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

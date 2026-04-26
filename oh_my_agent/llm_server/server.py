"""
本地 LLM HTTP 服务器：单进程加载 base + adapter，单端点切换推理。

Usage:
    # 带 adapter 启动
    conda run -n py312_t271_cuda python -m oh_my_agent.llm_server.server \
        --adapter models/webqsp/ablation/groupJ_schema_name --port 8788

    # 不加载 adapter（纯 base 模型）
    conda run -n py312_t271_cuda python -m oh_my_agent.llm_server.server \
        --port 8788

API:
    POST /generate
        body: {"prompt": "...", "use_adapter": true, "max_new_tokens": 256,
               "temperature": 0.0, "system_prompt": "..."}
        response: {"text": "...", "used_adapter": true, "tokens_generated": 42, "elapsed_ms": 123.4}

    GET  /health
    GET  /info
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from .app import create_app
from .config import ServerConfig
from .engine import ModelEngine
from .scheduler import BatchScheduler


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="本地 LLM HTTP 服务器（/generate 单端点切换 base / adapter）"
    )
    p.add_argument(
        "--model",
        default="unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
        help="base 模型名称或本地路径",
    )
    p.add_argument("--adapter", default=None, help="LoRA adapter 目录（不传则只提供 base 推理）")
    p.add_argument("--port", type=int, default=8788)
    p.add_argument("--host", default="0.0.0.0")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # transformers warning_once 用 % 格式但传了 FutureWarning 类作为参数，会触发 logging 内部异常
    logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

    cfg = ServerConfig.from_args_and_env(args)
    engine = ModelEngine()
    engine.load(cfg)
    scheduler = BatchScheduler(engine, cfg.max_batch_size, cfg.batch_wait_seconds)
    app = create_app(engine, scheduler)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()

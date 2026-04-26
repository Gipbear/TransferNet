"""LLM Server 配置，集中 argparse 参数与环境变量。"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ServerConfig:
    model_name: str
    adapter_path: Optional[str]
    host: str
    port: int
    max_batch_size: int        # 来自 LLM_SERVER_MAX_BATCH_SIZE env，默认 4，最小 1
    batch_wait_seconds: float  # 来自 LLM_SERVER_BATCH_WAIT_MS env（毫秒转秒），默认 20ms，最小 0.0
    max_seq_length: int = 2048

    @classmethod
    def from_args_and_env(cls, args: argparse.Namespace) -> "ServerConfig":
        """从 argparse.Namespace 与环境变量构造 ServerConfig。

        env 变量：
          LLM_SERVER_MAX_BATCH_SIZE  整数，批处理最大批大小，默认 4，强制 >= 1
          LLM_SERVER_BATCH_WAIT_MS   浮点毫秒，批收集等待时间，默认 20，转为秒后强制 >= 0.0
        """
        max_batch_size = max(1, int(os.environ.get("LLM_SERVER_MAX_BATCH_SIZE", "4")))
        batch_wait_seconds = max(0.0, float(os.environ.get("LLM_SERVER_BATCH_WAIT_MS", "20")) / 1000)
        return cls(
            model_name=args.model,
            adapter_path=args.adapter,
            host=args.host,
            port=args.port,
            max_batch_size=max_batch_size,
            batch_wait_seconds=batch_wait_seconds,
        )

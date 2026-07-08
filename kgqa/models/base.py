"""模型接口：加载 ckpt → 前向 → 中间得分（训练循环不并入）。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from kgqa.scores.base import ScoreBundle


class ScoreProducer(ABC):
    @abstractmethod
    def load_checkpoint(self, ckpt_path: str) -> None: ...

    @abstractmethod
    def produce(self, input_dir: str, qa_file: str, *, split: str = "test",
                batch_size: int = 16, topk: int = 500) -> ScoreBundle: ...

"""兼容路径：CWQ producer 已迁至 :mod:`kgqa.backbone.cwq`。"""
from kgqa.backbone.cwq import CWQScoreProducer, _read_vocab, _valid_lines

__all__ = ["CWQScoreProducer", "_read_vocab", "_valid_lines"]

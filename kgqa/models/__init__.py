"""兼容路径：在线得分生产器已迁至 :mod:`kgqa.backbone`。"""
from kgqa.backbone import make_score_producer

__all__ = ["make_score_producer"]

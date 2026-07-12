"""兼容路径：QA 解析工具已迁至 :mod:`kgqa.core.qa_formats`。"""
from kgqa.core.qa_formats import (
    WebQSPQASample,
    clean_question_text,
    load_webqsp_qa_samples,
    parse_webqsp_qa_line,
)

__all__ = [
    "WebQSPQASample", "clean_question_text", "load_webqsp_qa_samples",
    "parse_webqsp_qa_line",
]

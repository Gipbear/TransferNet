"""Hugging Face 本地缓存优先加载。"""
from __future__ import annotations

import logging
from typing import Any, TypeVar


T = TypeVar("T")
_LOG = logging.getLogger(__name__)

# 正常 BERT/BGE 词表在 3 万量级；只含特殊 token 的空壳约为 5
_MIN_VOCAB_SIZE = 1000


def _looks_degraded(obj: Any) -> bool:
    """判断是否为只含特殊 token 的空壳 tokenizer。

    transformers 5.x 在快照缺少 vocab.txt/tokenizer.json 时不再抛 OSError，
    而是静默返回一个词表只有几个特殊 token 的对象，会把所有词编码成 [UNK]。
    非 tokenizer 对象没有 vocab_size 属性，直接跳过。
    """
    vocab_size = getattr(obj, "vocab_size", None)
    return isinstance(vocab_size, int) and vocab_size < _MIN_VOCAB_SIZE


def from_pretrained_local_first(loader: Any, model_id: str, **kwargs: Any) -> T:
    """先只读本地缓存；缓存缺失或残缺时才允许联网下载。"""
    try:
        obj = loader.from_pretrained(model_id, local_files_only=True, **kwargs)
    except OSError:
        _LOG.info("本地 Hugging Face 缓存缺失，开始下载: %s", model_id)
        return loader.from_pretrained(model_id, local_files_only=False, **kwargs)

    if not _looks_degraded(obj):
        return obj

    _LOG.warning(
        "本地 Hugging Face 缓存的 tokenizer 词表异常(vocab_size=%s)，重新下载: %s",
        getattr(obj, "vocab_size", None), model_id,
    )
    obj = loader.from_pretrained(model_id, local_files_only=False, **kwargs)
    if _looks_degraded(obj):
        raise RuntimeError(
            f"{model_id} 的 tokenizer 词表仍然异常"
            f"(vocab_size={getattr(obj, 'vocab_size', None)})，"
            "请检查 Hugging Face 缓存是否完整"
        )
    return obj


def resolve_model_path_local_first(model_id: str) -> str:
    """优先返回本地快照目录，避免加载器为仓库名额外请求远端元数据。"""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(repo_id=model_id, local_files_only=True)
    except LocalEntryNotFoundError:
        _LOG.info("本地 Hugging Face 模型快照缺失，开始下载: %s", model_id)
        return snapshot_download(repo_id=model_id, local_files_only=False)

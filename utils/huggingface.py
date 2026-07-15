"""Hugging Face 本地缓存优先加载。"""
from __future__ import annotations

import logging
from typing import Any, TypeVar


T = TypeVar("T")
_LOG = logging.getLogger(__name__)


def from_pretrained_local_first(loader: Any, model_id: str, **kwargs: Any) -> T:
    """先只读本地缓存；缓存缺失时才允许联网下载。"""
    try:
        return loader.from_pretrained(model_id, local_files_only=True, **kwargs)
    except OSError:
        _LOG.info("本地 Hugging Face 缓存缺失，开始下载: %s", model_id)
        return loader.from_pretrained(model_id, local_files_only=False, **kwargs)


def resolve_model_path_local_first(model_id: str) -> str:
    """优先返回本地快照目录，避免加载器为仓库名额外请求远端元数据。"""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(repo_id=model_id, local_files_only=True)
    except LocalEntryNotFoundError:
        _LOG.info("本地 Hugging Face 模型快照缺失，开始下载: %s", model_id)
        return snapshot_download(repo_id=model_id, local_files_only=False)

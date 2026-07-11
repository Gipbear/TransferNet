"""实验目录 manifest:配置快照 + 上游文件指纹 + 断点续跑判定。

一个实验目录一个 manifest.json,build / train / eval 各占一节;
断点续跑判定 = 「产物存在 且 对应节的 config+inputs 一致」。
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime


def file_fingerprint(path: str) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": os.path.abspath(path),
            "sha256": h.hexdigest(),
            "size": os.path.getsize(path)}


def make_section(config: dict, inputs: dict[str, str]) -> dict:
    """config 须只含 JSON 原生类型(与磁盘往返后仍可比对相等)。"""
    return {
        "config": config,
        "inputs": {name: file_fingerprint(p) for name, p in inputs.items()},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def sections_compatible(a: dict, b: dict) -> bool:
    """断点续跑判定只看 config + inputs,忽略 stats/created_at。"""
    return (a.get("config") == b.get("config")
            and a.get("inputs") == b.get("inputs"))


def load(manifest_path: str) -> dict:
    if not os.path.isfile(manifest_path):
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def merge_section(manifest_path: str, name: str, section: dict) -> None:
    manifest = load(manifest_path)
    manifest[name] = section
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

"""统一记录一次 KGQA 实验运行的日志、进度和运行清单。

本模块只使用标准库。调用方将 ``--run_dir`` 指向实际实验目录后，会得到：

* ``run_manifest.json``：命令、代码版本、配置和上游输入指纹；
* ``progress.json``：可被脚本或界面轮询的轻量进度；
* ``logs/run.log`` 和 ``logs/events.jsonl``：人读日志与结构化事件。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def file_fingerprint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """返回输入文件的路径、大小和 SHA256，用于跨章节追溯。"""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": str(source.resolve()),
        "size": source.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """为现役 CLI 添加统一运行目录、日志与进度参数。"""
    parser.add_argument(
        "--run_dir", default="", help="实验运行目录；写入运行清单、日志和进度文件"
    )
    parser.add_argument("--log_level", default="INFO", help="日志级别，默认 INFO")
    parser.add_argument("--no_progress", action="store_true", help="关闭 tqdm 进度条")
    parser.add_argument(
        "--progress_interval", type=int, default=50,
        help="每处理多少条样本更新一次进度文件，默认 50",
    )


def _normalise_level(level: str) -> int:
    numeric = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"未知日志级别: {level}")
    return numeric


def configure_runtime(
    args: argparse.Namespace,
    *,
    command: str,
    fallback_run_dir: str | os.PathLike[str] | None = None,
    manifest: dict[str, Any] | None = None,
) -> Path | None:
    """初始化运行目录；未提供目录且没有回退目录时只配置控制台日志。"""
    raw_run_dir = getattr(args, "run_dir", "") or fallback_run_dir
    level = _normalise_level(getattr(args, "log_level", "INFO"))
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    run_dir: Path | None = None
    if raw_run_dir:
        run_dir = Path(raw_run_dir).resolve()
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs_dir / "run.log", encoding="utf-8")
        handlers.append(file_handler)
        run_manifest = {
            "schema_version": 1,
            "command": command,
            "argv": sys.argv[1:],
            "started_at": _now(),
            "git_commit": _git_commit(),
            "python": sys.version.split()[0],
            **(manifest or {}),
        }
        _atomic_json_write(run_dir / "run_manifest.json", run_manifest)
        update_progress(run_dir, status="running", phase=command)
        emit_event(run_dir, "phase_start", command=command)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return run_dir


def emit_event(run_dir: str | os.PathLike[str] | None, event: str, **fields: Any) -> None:
    """追加一条结构化事件；未启用运行目录时静默跳过。"""
    if not run_dir:
        return
    path = Path(run_dir) / "logs" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": _now(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_progress(
    run_dir: str | os.PathLike[str] | None,
    *,
    completed: int | None = None,
    total: int | None = None,
    status: str = "running",
    phase: str | None = None,
    **fields: Any,
) -> None:
    """原子更新进度。completed/total 缺省时适合记录阶段状态。"""
    if not run_dir:
        return
    payload: dict[str, Any] = {"updated_at": _now(), "status": status, **fields}
    if completed is not None:
        payload["completed"] = completed
    if total is not None:
        payload["total"] = total
    if phase is not None:
        payload["phase"] = phase
    _atomic_json_write(Path(run_dir) / "progress.json", payload)

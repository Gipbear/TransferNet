"""实验编排脚本共享的小型工具。"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(project_dir: Path, value: str) -> Path:
    """将配置中的相对路径按项目根目录解析。"""
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def command_text(command: Iterable[str]) -> str:
    """使用 JSON 风格展示命令，避免 shell 转义歧义。"""
    return " ".join(json.dumps(str(item), ensure_ascii=False) for item in command)


def run_command(command: list[str], run_dir: Path, *, dry_run: bool) -> None:
    """执行命令，实时转发输出，并同时存入统一控制台日志。"""
    text = command_text(command)
    logging.getLogger(__name__).info("执行命令: %s", text)
    console_path = run_dir / "logs" / "console.log"
    if dry_run:
        print(f"[演练] {text}")
        console_path.parent.mkdir(parents=True, exist_ok=True)
        with console_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[演练] {text}\n")
        return
    console_path.parent.mkdir(parents=True, exist_ok=True)
    with console_path.open("ab") as handle:
        handle.write(f"$ {text}\n".encode("utf-8"))
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        last_byte = b"\n"
        with process.stdout:
            while chunk := process.stdout.read1(8192):
                handle.write(chunk)
                if hasattr(sys.stdout, "buffer"):
                    sys.stdout.buffer.write(chunk)
                else:
                    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
                last_byte = chunk[-1:]
        returncode = process.wait()
        if last_byte != b"\n":
            handle.write(b"\n")
    if returncode:
        raise SystemExit(f"命令失败（退出码 {returncode}）: {text}")


def require_fields(config: dict[str, Any], *fields: str) -> None:
    # 空列表/空映射有时是合法的显式配置（例如只运行第五章回放消融），
    # 因而只拒绝字段缺失、null 或空字符串。
    missing = [
        field for field in fields
        if field not in config or config[field] is None or config[field] == ""
    ]
    if missing:
        raise ValueError(f"实验配置缺少必填字段: {', '.join(missing)}")

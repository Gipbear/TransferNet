"""提供独占运行锁、原子缓存写入与末行损坏恢复。"""

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


class ArtifactStore:
    @staticmethod
    def atomic_text(target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(target)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()

    @staticmethod
    @contextmanager
    def exclusive(target: Path) -> Iterator[None]:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"已有进程占用输出目录: {target.parent}") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    @contextmanager
    def cache_lock(target: Path) -> Iterator[None]:
        with target.open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @classmethod
    def recover_tail(cls, target: Path, offset: int, tail: bytes) -> None:
        with NamedTemporaryFile(dir=target.parent, prefix=target.name + ".torn.", delete=False) as backup:
            backup.write(tail)
        with target.open("r+b") as handle:
            handle.truncate(offset)

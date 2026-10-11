"""显式加载名称资源，并对实际解析的同一份字节计算指纹。"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EntityNameResource:
    names: dict[str, str]
    fingerprint: dict | None

    @classmethod
    def load(cls, path: str | Path | None) -> "EntityNameResource":
        if path is None:
            return cls({}, None)
        source = Path(path).resolve()
        content = source.read_bytes()
        names = json.loads(content)
        if not isinstance(names, dict):
            raise TypeError("名称资源必须是 MID 到名称的映射")
        for mid, name in names.items():
            if not isinstance(mid, str) or not isinstance(name, str) or not name.strip():
                raise ValueError("名称资源的 MID 和名称必须是非空字符串")
        fingerprint = {"path": str(source), "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        return cls(names, fingerprint)

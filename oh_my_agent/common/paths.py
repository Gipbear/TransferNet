"""Shared helpers for reading reasoning-path structures."""

from __future__ import annotations

from typing import Any


def tail_from_edges(path_edges: list[Any]) -> str:
    """Return the tail entity of the last edge, or '' for an empty path."""
    if not path_edges:
        return ""
    return str(path_edges[-1][-1])


def tail_from_path_dict(path_dict: dict[str, Any]) -> str:
    """Return the tail entity of a {'path': [...]} dict, or '' if absent."""
    return tail_from_edges(path_dict.get("path", []))

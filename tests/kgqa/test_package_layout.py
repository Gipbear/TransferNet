"""kgqa 顶层目录只保留当前能力域。"""
from __future__ import annotations

import unittest
from pathlib import Path


KGQA_ROOT = Path(__file__).resolve().parents[2] / "kgqa"


class TestPackageLayout(unittest.TestCase):
    def test_unprotected_compatibility_packages_are_removed(self):
        for relative in (
            "models",
            "eval",
            "cli",
            "server",
            "llm_server",
            "kg",
            "scores",
            "datasets",
            "types.py",
            "agent/demo_page",
        ):
            with self.subTest(relative=relative):
                self.assertFalse((KGQA_ROOT / relative).exists())

if __name__ == "__main__":
    unittest.main()

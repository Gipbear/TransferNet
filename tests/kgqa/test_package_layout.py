"""kgqa 顶层目录只保留当前能力域与受保护的 ReaRev 兼容层。"""
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
            "agent/demo_page",
        ):
            with self.subTest(relative=relative):
                self.assertFalse((KGQA_ROOT / relative).exists())

    def test_rearev_compatibility_surface_remains(self):
        for relative in ("datasets", "kg", "scores", "types.py"):
            with self.subTest(relative=relative):
                self.assertTrue((KGQA_ROOT / relative).exists())


if __name__ == "__main__":
    unittest.main()

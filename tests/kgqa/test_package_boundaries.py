"""静态守卫：新能力域不得重新引入 agent 反向依赖。"""
from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _imports_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


class TestPackageBoundaries(unittest.TestCase):
    def assert_no_imports(self, package: str, forbidden: set[str]) -> None:
        package_dir = PROJECT_ROOT / "kgqa" / package
        if not package_dir.exists():
            return

        violations: list[str] = []
        for source in package_dir.rglob("*.py"):
            for imported in _imports_in(source):
                if any(imported == root or imported.startswith(f"{root}.") for root in forbidden):
                    violations.append(f"{source.relative_to(PROJECT_ROOT)} -> {imported}")
        self.assertEqual([], violations)

    def test_retrieve_does_not_depend_on_agent(self):
        self.assert_no_imports("retrieve", {"kgqa.agent"})

    def test_future_core_is_domain_independent(self):
        self.assert_no_imports(
            "core",
            {
                "kgqa.backbone",
                "kgqa.retrieve",
                "kgqa.pfit",
                "kgqa.agent",
                "kgqa.serving",
            },
        )

    def test_future_backbone_does_not_depend_on_agent(self):
        self.assert_no_imports("backbone", {"kgqa.agent"})


if __name__ == "__main__":
    unittest.main()

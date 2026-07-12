"""KG relation-completion path restoration tests."""
import tempfile
import unittest
from pathlib import Path

from kgqa.agent.demo_page.kg_paths import KGPathResolver


class TestKGPathResolver(unittest.TestCase):
    def _resolver(self) -> KGPathResolver:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "relations.dict").write_text(
            "r.one\t0\n"
            "r.two\t1\n"
            "r.parent\t2\n"
            "r.parent_reverse\t3\n",
            encoding="utf-8",
        )
        (root / "train.txt").write_text(
            "m.topic\tr.one\tm.mid\n"
            "m.mid\tr.two\tm.tail1\n"
            "m.mid\tr.two\tm.tail2\n"
            "m.child\tr.parent\tm.topic\n",
            encoding="utf-8",
        )
        return KGPathResolver(root)

    def tearDown(self):
        tmp = getattr(self, "tmp", None)
        if tmp is not None:
            tmp.cleanup()

    def test_resolve_many_restores_two_hop_paths(self):
        resolver = self._resolver()
        paths = resolver.resolve_many("m.topic", ["r.one", "r.two"], ["m.tail1", "m.tail2"])
        self.assertEqual(
            paths["m.tail1"],
            [["m.topic", "r.one", "m.mid"], ["m.mid", "r.two", "m.tail1"]],
        )
        self.assertEqual(
            paths["m.tail2"],
            [["m.topic", "r.one", "m.mid"], ["m.mid", "r.two", "m.tail2"]],
        )

    def test_reverse_relation_uses_generated_reverse_edge(self):
        resolver = self._resolver()
        paths = resolver.resolve_many("m.topic", ["r.parent_reverse"], ["m.child"])
        self.assertEqual(paths["m.child"], [["m.topic", "r.parent_reverse", "m.child"]])


if __name__ == "__main__":
    unittest.main()

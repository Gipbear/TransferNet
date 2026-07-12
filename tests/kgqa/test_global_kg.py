import unittest
from kgqa.retrieve.graph.global_kg import GlobalKG


class TestGlobalKG(unittest.TestCase):
    def _kg(self):
        # 三元组 (subj, rel, obj)
        return GlobalKG.from_triples([[0, 100, 1], [0, 101, 2], [1, 100, 3]])

    def test_neighbors(self):
        kg = self._kg()
        self.assertCountEqual(kg.neighbors(0), [(100, 1), (101, 2)])
        self.assertEqual(kg.neighbors(1), [(100, 3)])
        self.assertEqual(kg.neighbors(999), [])

    def test_all_edges(self):
        kg = self._kg()
        self.assertCountEqual(
            list(kg.all_edges()), [(0, 100, 1), (0, 101, 2), (1, 100, 3)]
        )

    def test_valid_edges_dict_attr_matches_neighbors(self):
        kg = self._kg()
        self.assertEqual(kg.valid_edges_dict.get(0, []), kg.neighbors(0))


if __name__ == "__main__":
    unittest.main()

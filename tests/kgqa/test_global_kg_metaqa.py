import os
import tempfile
import unittest

import numpy as np

from kgqa.kg.global_kg import GlobalKG


class TestGlobalKGMetaQA(unittest.TestCase):
    def _write_npy_dir(self):
        d = tempfile.mkdtemp()
        # 两条三元组: (0)-[10]->(1), (1)-[11]->(2)；npy 形状 (T,2)，第 1 列是 id
        subj = np.array([[0, 0], [1, 1]])
        rel = np.array([[0, 10], [1, 11]])
        obj = np.array([[0, 1], [1, 2]])
        np.save(os.path.join(d, "Msubj.npy"), subj)
        np.save(os.path.join(d, "Mrel.npy"), rel)
        np.save(os.path.join(d, "Mobj.npy"), obj)
        return d

    def test_from_metaqa_npy_builds_edges(self):
        d = self._write_npy_dir()
        kg = GlobalKG.from_metaqa_npy(d)
        self.assertCountEqual(kg.neighbors(0), [(10, 1)])
        self.assertCountEqual(kg.neighbors(1), [(11, 2)])
        self.assertEqual(kg.neighbors(2), [])

    def test_no_reverse_edges_added(self):
        d = self._write_npy_dir()
        kg = GlobalKG.from_metaqa_npy(d)
        # 不应凭空生成 1->0 的反向边（MetaQA npy 已含所需双向边，不额外补）
        self.assertNotIn((10, 0), kg.neighbors(1))


if __name__ == "__main__":
    unittest.main()

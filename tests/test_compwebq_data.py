import os
import unittest

import numpy as np
import torch

from CompWebQ.data import (
    CACHE_LAYOUT, Dataset, SparseOneHot, _cached_datasets, as_int32, cache_path,
    collate, make_data_loader,
)
from utils.path_utils import filter_tensor


def _dataset():
    questions = [
        ([0], {"input_ids": torch.tensor([[1]])}, [1], [[0, 0, 1]], [0, 1]),
    ]
    return Dataset(questions, {"a": 0, "b": 1})


class TestCompWebQDataLoading(unittest.TestCase):
    def test_cache_payload_uses_datasets(self):
        dataset = _dataset()
        ent2id, rel2id, datasets, legacy = _cached_datasets(
            ({"a": 0}, {"r": 0}, dataset, dataset, dataset)
        )

        self.assertEqual(ent2id, {"a": 0})
        self.assertEqual(rel2id, {"r": 0})
        self.assertEqual(datasets, (dataset, dataset, dataset))
        self.assertFalse(legacy)

    def test_legacy_loader_cache_is_accepted(self):
        dataset = _dataset()
        legacy_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        _, _, datasets, legacy = _cached_datasets(
            ({"a": 0}, {"r": 0}, legacy_loader, legacy_loader, legacy_loader)
        )

        self.assertEqual(datasets, (dataset, dataset, dataset))
        self.assertTrue(legacy)

    def test_loader_options_are_applied_at_runtime(self):
        tokenizer = object()
        loader = make_data_loader(
            _dataset(), batch_size=1, training=True, num_workers=0,
            pin_memory=True, persistent_workers=False,
            ent2id={"a": 0}, rel2id={"r": 0}, tokenizer=tokenizer,
        )

        self.assertEqual(loader.batch_size, 1)
        self.assertTrue(loader.pin_memory)
        self.assertFalse(loader.persistent_workers)
        self.assertEqual(loader.id2ent, {0: "a"})
        self.assertEqual(loader.id2rel, {0: "r"})
        self.assertIs(loader.tokenizer, tokenizer)

    def test_collate_adds_flattened_triple_batch_indices(self):
        sample_a = _dataset()[0]
        sample_b = (
            torch.tensor([1]), {"input_ids": torch.tensor([[2]])},
            torch.tensor([0]), torch.tensor([[1, 0, 0], [0, 0, 1]]),
            torch.tensor([0, 1]),
        )

        batch = collate([sample_a, sample_b], num_ents=2)

        self.assertEqual(len(batch), 6)
        self.assertIs(batch[3][0], sample_a[3])
        self.assertIs(batch[3][1], sample_b[3])
        self.assertTrue(torch.equal(batch[0].dense(), torch.tensor([[1.0, 0.0], [0.0, 1.0]])))
        self.assertTrue(torch.equal(batch[2].dense(), torch.tensor([[0.0, 1.0], [1.0, 0.0]])))
        self.assertTrue(torch.equal(batch[5], torch.tensor([0, 1, 1])))


class TestCachePath(unittest.TestCase):
    def test_different_encoders_never_share_a_cache_file(self):
        """换 encoder 必须换缓存文件。

        缓存存的是 tokenization 结果；此前各 encoder 共用 'cache.pt'，换 encoder
        会静默读到上一个 tokenizer 的结果，不报错、只让指标莫名变差。
        """
        names = ['bert-base-cased', 'bert-base-uncased', 'roberta-base',
                 'BAAI/bge-base-en-v1.5']
        paths = [cache_path('/d', n) for n in names]

        self.assertEqual(len(set(paths)), len(names))
        for p in paths:
            self.assertNotIn('/', p[len('/d/'):])  # repo id 里的斜杠不能变成子目录

    def test_add_rev_still_separates(self):
        self.assertNotEqual(cache_path('/d', 'bert-base-cased', add_rev=True),
                            cache_path('/d', 'bert-base-cased', add_rev=False))

    def test_layout_version_is_in_the_filename(self):
        """存储布局变了必须换文件名，不能靠读进来再判断。

        v1 缓存里躺着的是 list[list[int]] 版的三元组，CWQ 全量 25.8 GB；等加载完
        再发现版本不对，内存早就炸了。文件名带版本才能拦在 pickle.load 之前。
        """
        p = cache_path('/d', 'bert-base-cased')
        self.assertIn(CACHE_LAYOUT, os.path.basename(p))
        self.assertNotEqual(p, '/d/cache_bert-base-cased.pt')  # v1 的名字


class TestInt32Storage(unittest.TestCase):
    """id 序列改用 int32 数组存，取出来的东西必须跟 list 存法逐位相同。"""

    def _questions(self, wrap):
        return [(
            wrap([0, 1]), {"input_ids": torch.tensor([[1]])}, wrap([1]),
            wrap([[0, 0, 1], [1, 0, 0]]), wrap([0, 1]),
        )]

    def test_int32_storage_yields_identical_samples(self):
        as_list = Dataset(self._questions(list), {"a": 0, "b": 1})[0]
        as_arr = Dataset(self._questions(as_int32), {"a": 0, "b": 1})[0]

        for i in (0, 2, 3, 4):
            self.assertEqual(as_arr[i].dtype, torch.long)  # 下游还是按 long 用
            self.assertTrue(torch.equal(as_list[i], as_arr[i]))

    def test_int32_storage_is_much_smaller(self):
        triples = [[i, 0, i + 1] for i in range(2000)]

        packed = as_int32(triples)

        self.assertEqual(packed.dtype, np.int32)
        self.assertEqual(packed.shape, (2000, 3))
        # list[list[int]] 每条约 187 B；int32 是 12 B，一个数量级以上的差
        self.assertLess(packed.nbytes / len(triples), 20)


class TestSparseOneHot(unittest.TestCase):
    """稠密 one-hot 推迟到目标设备上展开，语义须与原先的 CPU 稠密张量逐位一致。"""

    def _sample(self):
        return SparseOneHot.from_rows(
            [torch.tensor([2, 0]), torch.tensor([]).long(), torch.tensor([1])],
            num_ents=4,
        )

    def test_dense_matches_scatter_semantics(self):
        expected = torch.tensor([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        self.assertTrue(torch.equal(self._sample().dense(), expected))

    def test_getitem_returns_sorted_column_indices(self):
        one_hot = self._sample()

        self.assertEqual(one_hot[0].tolist(), [0, 2])
        self.assertEqual(one_hot[1].tolist(), [])
        self.assertEqual(len(one_hot), 3)

    def test_gather_rows_matches_dense_gather(self):
        one_hot = self._sample()
        cols = torch.tensor([2, 0, 3])

        expected = one_hot.dense().gather(1, cols.unsqueeze(-1)).squeeze(1)
        self.assertTrue(torch.equal(one_hot.gather_rows(cols), expected))
        self.assertEqual(one_hot.gather_rows(cols).tolist(), [1.0, 0.0, 0.0])

    def test_matches_legacy_cpu_dense_construction(self):
        """与被替换掉的 CPU 稠密实现随机对拍，确保三个消费点语义不变。"""

        def legacy_batch_one_hot(rows, num_ents):
            sizes = torch.tensor([row.shape[0] for row in rows])
            batch_idx = torch.repeat_interleave(torch.arange(len(rows)), sizes)
            dense = torch.zeros(len(rows), num_ents)
            dense[batch_idx, torch.cat(rows)] = 1
            return dense

        torch.manual_seed(0)
        num_ents = 97
        for _ in range(20):
            rows = [torch.randperm(num_ents)[:torch.randint(0, 6, (1,)).item()]
                    for _ in range(5)]
            legacy = legacy_batch_one_hot(rows, num_ents)
            one_hot = SparseOneHot.from_rows(rows, num_ents)

            # model.forward 的展开点
            self.assertTrue(torch.equal(one_hot.dense(), legacy))
            # predict.validate 的 top-1 命中判定
            cols = torch.randint(0, num_ents, (5,))
            self.assertTrue(torch.equal(
                one_hot.gather_rows(cols),
                legacy.gather(1, cols.unsqueeze(-1)).squeeze(1)))
            # predict/dump_scores/kgqa backbone 的 topic、gold 取用
            for row in range(5):
                self.assertEqual(
                    one_hot[row].tolist(),
                    [x for (x, _) in filter_tensor(legacy[row], 0.5)])


if __name__ == "__main__":
    unittest.main()

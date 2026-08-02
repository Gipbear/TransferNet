import unittest

import torch

from CompWebQ.data import (
    Dataset, SparseOneHot, _cached_datasets, collate, make_data_loader,
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

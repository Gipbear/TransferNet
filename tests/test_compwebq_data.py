import unittest

import torch

from CompWebQ.data import Dataset, _cached_datasets, collate, make_data_loader


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
        self.assertTrue(torch.equal(batch[0], torch.tensor([[1.0, 0.0], [0.0, 1.0]])))
        self.assertTrue(torch.equal(batch[2], torch.tensor([[0.0, 1.0], [1.0, 0.0]])))
        self.assertTrue(torch.equal(batch[5], torch.tensor([0, 1, 1])))


if __name__ == "__main__":
    unittest.main()

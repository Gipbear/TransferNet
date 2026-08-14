"""CompWebQ dev 并入 train 的切分逻辑测试。

容量饱和分析指向训练样本量是瓶颈（22089 条）；dev 有 3519 条可并入。
合并后必须留一小块 dev 作验证集，否则只能按 test 选 epoch，等于测试集泄漏。
"""
import unittest

from CompWebQ.data import Dataset, merge_dev_into_train


def _fake_dataset(n: int, offset: int = 0) -> Dataset:
    """questions 元素只要能被 list 拼接即可，这里用可辨识的整数占位。"""
    return Dataset([offset + i for i in range(n)], ent2id={})


class TestMergeDevIntoTrain(unittest.TestCase):
    def test_holdout_size_and_total_preserved(self):
        train, dev = _fake_dataset(100), _fake_dataset(30, offset=1000)
        new_train, val = merge_dev_into_train(train, dev, holdout=10)
        self.assertEqual(len(val), 10)
        self.assertEqual(len(new_train), 120)
        self.assertEqual(len(new_train) + len(val), 130)

    def test_val_and_train_are_disjoint(self):
        train, dev = _fake_dataset(100), _fake_dataset(30, offset=1000)
        new_train, val = merge_dev_into_train(train, dev, holdout=10)
        self.assertEqual(set(val.questions) & set(new_train.questions), set())

    def test_val_comes_from_dev_head(self):
        train, dev = _fake_dataset(100), _fake_dataset(30, offset=1000)
        _, val = merge_dev_into_train(train, dev, holdout=10)
        self.assertEqual(val.questions, list(range(1000, 1010)))

    def test_train_keeps_original_order_then_dev_tail(self):
        train, dev = _fake_dataset(5), _fake_dataset(4, offset=1000)
        new_train, _ = merge_dev_into_train(train, dev, holdout=2)
        self.assertEqual(new_train.questions, [0, 1, 2, 3, 4, 1002, 1003])

    def test_ent2id_inherited_from_train(self):
        train = Dataset([1, 2, 3], ent2id={"a": 0})
        dev = Dataset([4, 5], ent2id={"a": 0})
        new_train, val = merge_dev_into_train(train, dev, holdout=1)
        self.assertEqual(new_train.ent2id, {"a": 0})
        self.assertEqual(val.ent2id, {"a": 0})

    def test_holdout_zero_rejected(self):
        """holdout=0 会让验证集为空，只能按 test 选 epoch，必须挡住。"""
        train, dev = _fake_dataset(10), _fake_dataset(5)
        with self.assertRaises(ValueError):
            merge_dev_into_train(train, dev, holdout=0)

    def test_holdout_not_smaller_than_dev_rejected(self):
        train, dev = _fake_dataset(10), _fake_dataset(5)
        with self.assertRaises(ValueError):
            merge_dev_into_train(train, dev, holdout=5)

    def test_original_datasets_untouched(self):
        train, dev = _fake_dataset(5), _fake_dataset(4, offset=1000)
        merge_dev_into_train(train, dev, holdout=2)
        self.assertEqual(len(train), 5)
        self.assertEqual(len(dev), 4)


if __name__ == "__main__":
    unittest.main()

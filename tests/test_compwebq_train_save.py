"""checkpoint 落盘策略测试。

原实现 --save_best_only 是「刷新 test 纪录就存」,而 test acc 前期单调上升,
等于每个 epoch 都存一个 490MB 文件(实测 row 实验 9 个 epoch 占 4.1G)。
改为:best 只保留最新一个(旧的删掉),另按 --save_every 周期留快照。
"""
import unittest

from CompWebQ.train import should_save_checkpoint


class TestShouldSaveCheckpoint(unittest.TestCase):
    def test_save_all_when_best_only_disabled(self):
        """未开 --save_best_only 时保持原行为:每次评估都落盘。"""
        do_save, is_best = should_save_checkpoint(
            epoch=3, test_acc=0.1, best_test_acc=0.9, save_best_only=False, save_every=5
        )
        self.assertTrue(do_save)
        self.assertFalse(is_best)

    def test_new_record_always_saved_and_flagged_best(self):
        do_save, is_best = should_save_checkpoint(
            epoch=3, test_acc=0.50, best_test_acc=0.48, save_best_only=True, save_every=5
        )
        self.assertTrue(do_save)
        self.assertTrue(is_best)

    def test_no_record_and_off_cycle_skips(self):
        do_save, is_best = should_save_checkpoint(
            epoch=3, test_acc=0.47, best_test_acc=0.48, save_best_only=True, save_every=5
        )
        self.assertFalse(do_save)
        self.assertFalse(is_best)

    def test_periodic_snapshot_saved_without_record(self):
        # epoch 从 0 计,第 5 个 epoch 即 epoch=4
        do_save, is_best = should_save_checkpoint(
            epoch=4, test_acc=0.47, best_test_acc=0.48, save_best_only=True, save_every=5
        )
        self.assertTrue(do_save)
        self.assertFalse(is_best, "周期快照不是 best,不能触发删除旧 best")

    def test_save_every_zero_disables_periodic(self):
        do_save, _ = should_save_checkpoint(
            epoch=4, test_acc=0.47, best_test_acc=0.48, save_best_only=True, save_every=0
        )
        self.assertFalse(do_save)

    def test_record_on_cycle_boundary_is_best(self):
        """既刷新纪录又踩在周期上时按 best 处理,否则旧 best 不会被清理。"""
        do_save, is_best = should_save_checkpoint(
            epoch=4, test_acc=0.50, best_test_acc=0.48, save_best_only=True, save_every=5
        )
        self.assertTrue(do_save)
        self.assertTrue(is_best)

    def test_first_epoch_is_record_against_sentinel(self):
        do_save, is_best = should_save_checkpoint(
            epoch=0, test_acc=0.01, best_test_acc=-1.0, save_best_only=True, save_every=5
        )
        self.assertTrue(do_save)
        self.assertTrue(is_best)


if __name__ == "__main__":
    unittest.main()

"""pfit.train 训练配置构造(免 GPU):显存峰值控制 + checkpoint 保护。

回归背景:cwq/prompt_sft/dont_v2 在 epoch 1 的 eval 之后速度掉到十分之一
(功耗 52W/165W、显存 16013/16380MiB),诊断为 eval 阶段 logits 峰值把显存顶满、
allocator 碎片化后落入 sysmem fallback;且 save_strategy="no" 导致中断即全损。
"""
import os
import unittest
from unittest import mock


def _args(**overrides):
    from kgqa.pfit.train import build_training_args

    kwargs = {
        "adapter_dir": "/tmp/adapter",
        "epochs": 2,
        "batch_size": 4,
        "grad_accum": 8,
        "lr": 2e-4,
        "warmup_ratio": 0.05,
        "max_seq_len": 1280,
        "seed": 17,
        "bf16": True,
        "has_eval": True,
    }
    kwargs.update(overrides)
    return build_training_args(**kwargs)


class TestBuildTrainingArgs(unittest.TestCase):
    def test_eval_batch_is_one(self):
        # eval logits [bs, seq, vocab] 是显存断崖主因,eval batch 必须与 train batch 解耦
        self.assertEqual(_args().per_device_eval_batch_size, 1)

    def test_checkpoint_saved_per_epoch(self):
        # transformers 会把字符串转成 SaveStrategy/IntervalStrategy 枚举
        args = _args()
        self.assertEqual(args.save_strategy.value, "epoch")
        self.assertGreaterEqual(args.save_total_limit, 1)

    def test_eval_strategy_follows_has_eval(self):
        self.assertEqual(_args(has_eval=True).eval_strategy.value, "epoch")
        self.assertEqual(_args(has_eval=False).eval_strategy.value, "no")

    def test_train_hparams_passthrough(self):
        # 与已完成的 14 组 ch4 实验保持同一超参,修复只动工程侧
        args = _args()
        self.assertEqual(args.per_device_train_batch_size, 4)
        self.assertEqual(args.gradient_accumulation_steps, 8)
        self.assertEqual(args.num_train_epochs, 2)
        self.assertAlmostEqual(args.learning_rate, 2e-4)
        self.assertAlmostEqual(args.warmup_ratio, 0.05)
        self.assertEqual(args.seed, 17)

    def test_precision_follows_bf16_flag(self):
        self.assertTrue(_args(bf16=True).bf16)
        self.assertFalse(_args(bf16=True).fp16)
        self.assertTrue(_args(bf16=False).fp16)
        self.assertFalse(_args(bf16=False).bf16)


class TestMemoryCleanupCallback(unittest.TestCase):
    def test_empties_cuda_cache_after_eval(self):
        from kgqa.pfit.train import make_memory_cleanup_callback

        cb = make_memory_cleanup_callback()
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.empty_cache") as empty:
            cb.on_evaluate(None, None, None)
        empty.assert_called_once()

    def test_noop_without_cuda(self):
        from kgqa.pfit.train import make_memory_cleanup_callback

        cb = make_memory_cleanup_callback()
        with mock.patch("torch.cuda.is_available", return_value=False), \
             mock.patch("torch.cuda.empty_cache") as empty:
            cb.on_evaluate(None, None, None)
        empty.assert_not_called()


class TestAllocatorEnv(unittest.TestCase):
    def test_expandable_segments_configured(self):
        import kgqa.pfit.train  # noqa: F401  导入即应设好 allocator 环境变量

        self.assertTrue(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))


if __name__ == "__main__":
    unittest.main()

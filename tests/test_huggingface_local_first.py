import unittest
from types import SimpleNamespace

from utils.huggingface import from_pretrained_local_first


class _Loader:
    calls = []
    fail_local = False

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append((model_id, kwargs))
        if cls.fail_local and kwargs["local_files_only"]:
            raise OSError("本地缓存缺失")
        return {"model_id": model_id, **kwargs}


class TestLocalFirstLoading(unittest.TestCase):
    def setUp(self):
        _Loader.calls = []
        _Loader.fail_local = False

    def test_uses_local_cache_without_network_fallback(self):
        result = from_pretrained_local_first(_Loader, "repo/model", return_dict=True)

        self.assertTrue(result["local_files_only"])
        self.assertEqual(_Loader.calls, [("repo/model", {"local_files_only": True, "return_dict": True})])

    def test_retries_online_only_after_local_cache_is_missing(self):
        _Loader.fail_local = True
        result = from_pretrained_local_first(_Loader, "repo/model")

        self.assertFalse(result["local_files_only"])
        self.assertEqual(_Loader.calls, [
            ("repo/model", {"local_files_only": True}),
            ("repo/model", {"local_files_only": False}),
        ])


class _VocabLoader:
    """按调用顺序返回预置词表大小的 tokenizer 桩。"""

    def __init__(self, *vocab_sizes):
        self.vocab_sizes = list(vocab_sizes)
        self.calls = []

    def from_pretrained(self, model_id, **kwargs):
        self.calls.append(kwargs["local_files_only"])
        return SimpleNamespace(vocab_size=self.vocab_sizes.pop(0))


class TestDegradedVocabGuard(unittest.TestCase):
    """transformers 5.x 缺 vocab 文件时不再抛 OSError，而是静默返回空壳 tokenizer。"""

    def test_healthy_vocab_skips_redownload(self):
        loader = _VocabLoader(28996)

        from_pretrained_local_first(loader, "bert-base-cased")

        self.assertEqual(loader.calls, [True])

    def test_degraded_vocab_triggers_redownload(self):
        loader = _VocabLoader(5, 28996)

        result = from_pretrained_local_first(loader, "bert-base-cased")

        self.assertEqual(result.vocab_size, 28996)
        self.assertEqual(loader.calls, [True, False])

    def test_still_degraded_after_redownload_raises(self):
        loader = _VocabLoader(5, 5)

        with self.assertRaises(RuntimeError) as ctx:
            from_pretrained_local_first(loader, "bert-base-cased")

        self.assertIn("词表仍然异常", str(ctx.exception))

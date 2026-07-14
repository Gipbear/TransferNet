import unittest

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

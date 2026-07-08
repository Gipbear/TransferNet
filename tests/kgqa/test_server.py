import unittest
from kgqa.types import RetrieveResult


class _StubBackend:
    class _B:  # 模拟 bundle.samples 长度
        samples = [None, None, None]
    bundle = _B()

    def retrieve(self, sample_index, **params):
        return RetrieveResult(question="q", topics=["m.t"], hop=1,
                              paths=[{"path": [["m.t", "r", "m.a"]], "log_score": -0.1}],
                              prediction={"m.a": 0.9}, elapsed_ms=0.5, sample_index=sample_index)


class TestServer(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from kgqa.server.path_retrieve_server import create_app
        self.client = TestClient(create_app(_StubBackend()))

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["n"], 3)

    def test_retrieve(self):
        resp = self.client.post("/retrieve", json={"sample_index": 2, "beam_size": 10})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["sample_index"], 2)
        self.assertEqual(body["paths"][0]["log_score"], -0.1)


if __name__ == "__main__":
    unittest.main()

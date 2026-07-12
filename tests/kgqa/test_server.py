import unittest
from pydantic import ValidationError
from kgqa.core.contracts import RetrieveResult


class _StubBackend:
    class _B:  # 模拟 bundle.samples 长度
        samples = [None, None, None]
    bundle = _B()

    def retrieve(self, sample_index, **params):
        self.last_params = params
        return RetrieveResult(question="q", topics=["m.t"], hop=1,
                              paths=[{"path": [["m.t", "r", "m.a"]], "log_score": -0.1}],
                              prediction={"m.a": 0.9}, elapsed_ms=0.5, sample_index=sample_index)


class TestServer(unittest.TestCase):
    def setUp(self):
        from kgqa.retrieve.api.path_retrieve_server import RetrieveRequest, create_app
        self.backend = _StubBackend()
        self.request_type = RetrieveRequest
        app = create_app(self.backend)
        self.endpoints = {
            route.path: route.endpoint
            for route in app.routes
            if hasattr(route, "endpoint")
        }

    def test_health(self):
        self.assertEqual(self.endpoints["/health"]()["n"], 3)

    def test_retrieve(self):
        body = self.endpoints["/retrieve"](
            self.request_type(sample_index=2, beam_size=10)
        )
        self.assertEqual(body["sample_index"], 2)
        self.assertEqual(body["paths"][0]["log_score"], -0.1)
        self.assertNotIn("method", self.backend.last_params)

    def test_retrieve_rejects_removed_method_parameter(self):
        with self.assertRaises(ValidationError):
            self.request_type(sample_index=0, method="baseline")


if __name__ == "__main__":
    unittest.main()

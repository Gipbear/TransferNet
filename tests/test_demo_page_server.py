"""demo_page FastAPI 接口测试（依赖注入 stub，不触网络/大数据）。"""
import unittest

from fastapi import HTTPException
from pydantic import ValidationError

from kgqa.agent.common.qa_data import WebQSPQASample
from kgqa.agent.web.data import QuestionIndex
from kgqa.agent.web.schema import ReplayIn, RetrieveIn
from kgqa.agent.web.server import create_app


def _questions():
    return QuestionIndex([
        WebQSPQASample(question_raw="q0 [m.0x]", question="what does jamaican people speak",
                       topic_mid="m.0x", gold_mids=["m.0g"]),
    ])


def _stub_retrieve(question=None, *, sample_index=None, eta=1.0,
                   beam_size=50, lambda_val=0.2, **kw):
    class R:
        named_mmr_reason_paths = [
            {"path": [["Jamaica", "r.lang", "Jamaican English"]], "log_score": -7.9}]
        named_topics = ["Jamaica"]
        named_prediction = {"Jamaican English": 0.99}
        elapsed_ms = 3.2
    return R()


class _StubReplayer:
    def replay(self, sample_index, **overrides):
        if sample_index != 0:
            raise KeyError(sample_index)
        return {"iterations": [], "final_answers": [], "stop_reason": "mixed_ratio",
                "calibration": {"dropped_answers": [], "relation_expanded_path_ids": [],
                                "group_expanded_names": []},
                "graph": {"nodes": [], "edges": [], "paths": []}}


class TestDemoPageAPI(unittest.TestCase):
    def setUp(self):
        self.app = create_app(questions=_questions(), retrieve_fn=_stub_retrieve,
                              replayer=_StubReplayer())

    def _endpoint(self, path):
        return next(route.endpoint for route in self.app.routes if route.path == path)

    def test_questions_search(self):
        body = self._endpoint("/api/questions")(q="jamaican", limit=20)
        self.assertEqual(body[0]["sample_index"], 0)

    def test_retrieve_final_config_flag(self):
        retrieve = self._endpoint("/api/retrieve")
        body = retrieve(RetrieveIn(
            sample_index=0, beam_size=50, lambda_val=0.2, eta=1.0))
        self.assertTrue(body["is_final_config"])
        self.assertEqual(body["graph"]["paths"][0]["id"], 1)
        body2 = retrieve(RetrieveIn(
            sample_index=0, beam_size=20, lambda_val=0.2, eta=1.0))
        self.assertFalse(body2["is_final_config"])

    def test_retrieve_rejects_alpha_final(self):
        with self.assertRaises(ValidationError):
            RetrieveIn(sample_index=0, alpha_final=1.0)

    def test_replay_passthrough(self):
        body = self._endpoint("/api/replay")(ReplayIn(sample_index=0))
        self.assertEqual(body["stop_reason"], "mixed_ratio")

    def test_retrieve_unknown_sample_404(self):
        with self.assertRaises(HTTPException) as ctx:
            self._endpoint("/api/retrieve")(RetrieveIn(
                sample_index=999, beam_size=50, lambda_val=0.2, eta=1.0))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_replay_unknown_sample_404(self):
        with self.assertRaises(HTTPException) as ctx:
            self._endpoint("/api/replay")(ReplayIn(sample_index=999))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_retrieve_backend_down_502(self):
        import requests as _requests
        def _down(**kw):
            raise _requests.exceptions.ConnectionError("refused")
        self.app = create_app(questions=_questions(), retrieve_fn=_down,
                              replayer=_StubReplayer())
        with self.assertRaises(HTTPException) as ctx:
            self._endpoint("/api/retrieve")(RetrieveIn(
                sample_index=0, beam_size=50, lambda_val=0.2, eta=1.0))
        self.assertEqual(ctx.exception.status_code, 502)


if __name__ == "__main__":
    unittest.main()

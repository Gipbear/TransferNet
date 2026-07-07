"""demo_page FastAPI 接口测试（依赖注入 stub，不触网络/大数据）。"""
import unittest

from fastapi.testclient import TestClient

from oh_my_agent.common.qa_data import WebQSPQASample
from oh_my_agent.demo_page.data import QuestionIndex
from oh_my_agent.demo_page.server import create_app


def _questions():
    return QuestionIndex([
        WebQSPQASample(question_raw="q0 [m.0x]", question="what does jamaican people speak",
                       topic_mid="m.0x", gold_mids=["m.0g"]),
    ])


def _stub_retrieve(question=None, *, sample_index=None, alpha_final=1.0,
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
        app = create_app(questions=_questions(), retrieve_fn=_stub_retrieve,
                         replayer=_StubReplayer())
        self.client = TestClient(app)

    def test_questions_search(self):
        resp = self.client.get("/api/questions", params={"q": "jamaican"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()[0]["sample_index"], 0)

    def test_retrieve_final_config_flag(self):
        resp = self.client.post("/api/retrieve", json={
            "sample_index": 0, "beam_size": 50, "lambda_val": 0.2, "alpha_final": 1.0})
        body = resp.json()
        self.assertTrue(body["is_final_config"])
        self.assertEqual(body["graph"]["paths"][0]["id"], 1)
        resp2 = self.client.post("/api/retrieve", json={
            "sample_index": 0, "beam_size": 20, "lambda_val": 0.2, "alpha_final": 1.0})
        self.assertFalse(resp2.json()["is_final_config"])

    def test_replay_passthrough(self):
        resp = self.client.post("/api/replay", json={"sample_index": 0})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["stop_reason"], "mixed_ratio")

    def test_retrieve_unknown_sample_404(self):
        resp = self.client.post("/api/retrieve", json={
            "sample_index": 999, "beam_size": 50, "lambda_val": 0.2, "alpha_final": 1.0})
        self.assertEqual(resp.status_code, 404)

    def test_replay_unknown_sample_404(self):
        resp = self.client.post("/api/replay", json={"sample_index": 999})
        self.assertEqual(resp.status_code, 404)

    def test_retrieve_backend_down_502(self):
        import requests as _requests
        def _down(**kw):
            raise _requests.exceptions.ConnectionError("refused")
        app = create_app(questions=_questions(), retrieve_fn=_down,
                         replayer=_StubReplayer())
        client = TestClient(app)
        resp = client.post("/api/retrieve", json={
            "sample_index": 0, "beam_size": 50, "lambda_val": 0.2, "alpha_final": 1.0})
        self.assertEqual(resp.status_code, 502)


if __name__ == "__main__":
    unittest.main()

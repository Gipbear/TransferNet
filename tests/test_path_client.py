import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.path_server.client import PathRetrievalClient


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "question": "who was vice president after kennedy died",
            "topics": ["m.0d3k14"],
            "hop": 2,
            "mmr_reason_paths": [],
            "prediction": {},
            "elapsed_ms": 1.0,
        }


class FakeStatusResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class PathRetrievalClientTests(unittest.TestCase):
    def test_retrieve_posts_mmr_parameters(self):
        client = PathRetrievalClient("http://localhost:8787")

        with patch(
            "oh_my_agent.path_server.client.requests.post",
            return_value=FakeResponse(),
        ) as post:
            resp = client.retrieve(
                "who was vice president after kennedy died",
                topic_entities=["m.0d3k14"],
                hop=2,
                beam_size=20,
                lambda_val=0.2,
                prediction_threshold=0.8,
            )

        self.assertEqual(resp.hop, 2)
        post.assert_called_once_with(
            "http://localhost:8787/retrieve",
            json={
                "question": "who was vice president after kennedy died",
                "topic_entities": ["m.0d3k14"],
                "hop": 2,
                "beam_size": 20,
                "lambda_val": 0.2,
                "prediction_threshold": 0.8,
            },
            timeout=120,
        )

    def test_health_uses_configured_timeout(self):
        client = PathRetrievalClient("http://localhost:8787", timeout=7)

        with patch(
            "oh_my_agent.path_server.client.requests.get",
            return_value=FakeStatusResponse({"status": "ok"}),
        ) as get:
            self.assertEqual(client.health(), {"status": "ok"})

        get.assert_called_once_with("http://localhost:8787/health", timeout=7)

    def test_info_uses_configured_timeout(self):
        client = PathRetrievalClient("http://localhost:8787", timeout=7)

        with patch(
            "oh_my_agent.path_server.client.requests.get",
            return_value=FakeStatusResponse({"model_loaded": True}),
        ) as get:
            self.assertEqual(client.info(), {"model_loaded": True})

        get.assert_called_once_with("http://localhost:8787/info", timeout=7)


if __name__ == "__main__":
    unittest.main()

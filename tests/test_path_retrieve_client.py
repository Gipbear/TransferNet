import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kgqa.retrieve.api.client import PathRetrieveClient


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "question": "[CLS] what does jamaican people speak [SEP]",
            "sample_index": 0,
            "topics": ["m.03_r3"],
            "hop": 1,
            "mmr_reason_paths": [],
            "prediction": {},
            "elapsed_ms": 1.0,
            "eta": 1.0,
            "step_score_mode": "joint",
            "threshold": 0.01,
            "beam_size": 50,
            "lambda_val": 0.5,
            "penalty_mode": "adaptive",
            "cache_path": "cache.pt",
        }


class FakeStatusResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class PathRetrieveClientTests(unittest.TestCase):
    def test_retrieve_posts_default_parameters(self):
        client = PathRetrieveClient("http://localhost:8789")

        with patch(
            "kgqa.retrieve.api.client.requests.post",
            return_value=FakeResponse(),
        ) as post:
            resp = client.retrieve("what does jamaican people speak", topic_entities=["m.03_r3"])

        self.assertEqual(resp.beam_size, 50)
        post.assert_called_once_with(
            "http://localhost:8789/retrieve",
            json={
                "question": "what does jamaican people speak",
                "sample_index": None,
                "topic_entities": ["m.03_r3"],
                "eta": 1.0,
                "step_score_mode": "joint",
                "threshold": 0.01,
                "beam_size": 50,
                "lambda_val": 0.2,
                "penalty_mode": "adaptive",
            },
            timeout=120,
        )

    def test_status_calls_use_configured_timeout(self):
        client = PathRetrieveClient("http://localhost:8789", timeout=7)

        with patch(
            "kgqa.retrieve.api.client.requests.get",
            return_value=FakeStatusResponse({"status": "ok"}),
        ) as get:
            self.assertEqual(client.health(), {"status": "ok"})

        get.assert_called_once_with("http://localhost:8789/health", timeout=7)

    def test_response_rejects_alpha_final(self):
        from kgqa.retrieve.api.client import PathRetrieveResponse

        with self.assertRaises(ValidationError):
            PathRetrieveResponse(**{
                **FakeResponse().json(),
                "alpha_final": 1.0,
            })


if __name__ == "__main__":
    unittest.main()

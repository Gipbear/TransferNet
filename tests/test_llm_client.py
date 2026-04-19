import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.llm_server.client import LLMClient


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "text": "ok",
            "used_adapter": False,
            "tokens_generated": 1,
            "elapsed_ms": 1.0,
        }


class LLMClientTests(unittest.TestCase):
    def test_generate_can_disable_adapter(self):
        client = LLMClient("http://localhost:8788")

        with patch("oh_my_agent.llm_server.client.requests.post", return_value=FakeResponse()) as post:
            client.generate(
                "hello",
                use_adapter=False,
                max_new_tokens=8,
                temperature=0.1,
                system_prompt="sys",
            )

        post.assert_called_once_with(
            "http://localhost:8788/generate",
            json={
                "prompt": "hello",
                "use_adapter": False,
                "max_new_tokens": 8,
                "temperature": 0.1,
                "system_prompt": "sys",
            },
            timeout=120,
        )

    def test_generate_can_enable_adapter(self):
        client = LLMClient("http://localhost:8788")

        with patch("oh_my_agent.llm_server.client.requests.post", return_value=FakeResponse()) as post:
            client.generate("hello", use_adapter=True)

        post.assert_called_once_with(
            "http://localhost:8788/generate",
            json={
                "prompt": "hello",
                "use_adapter": True,
                "max_new_tokens": 256,
                "temperature": 0.0,
                "system_prompt": None,
            },
            timeout=120,
        )


if __name__ == "__main__":
    unittest.main()

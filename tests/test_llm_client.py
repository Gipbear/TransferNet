import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kgqa.serving.llm.client import LLMClient, OpenAICompatibleLLMClient, SiliconFlowLLMClient


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


class FakeOpenAIResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 3},
        }


class FakeRateLimitResponse:
    status_code = 429

    def raise_for_status(self):
        import requests

        response = requests.Response()
        response.status_code = 429
        raise requests.HTTPError(response=response)


class LLMClientTests(unittest.TestCase):
    def test_generate_can_disable_adapter(self):
        client = LLMClient("http://localhost:8788")

        with patch("kgqa.serving.llm.client.requests.post", return_value=FakeResponse()) as post:
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

    def test_openai_compatible_client_uses_chat_completions(self):
        client = OpenAICompatibleLLMClient(
            "http://localhost:8788/v1",
            model="webqsp-agent",
            api_key="token",
        )

        with patch("kgqa.serving.llm.client.requests.post", return_value=FakeOpenAIResponse()) as post:
            response = client.generate(
                "hello",
                use_adapter=False,
                max_new_tokens=8,
                temperature=0.1,
                system_prompt="sys",
            )

        self.assertEqual(response.text, "ok")
        self.assertEqual(response.tokens_generated, 3)
        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args[0], "http://localhost:8788/v1/chat/completions")
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer token"})
        self.assertEqual(
            kwargs["json"],
            {
                "model": "webqsp-agent",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hello"},
                ],
                "max_tokens": 8,
                "temperature": 0.1,
            },
        )

    def test_openai_compatible_client_merges_extra_body(self):
        client = OpenAICompatibleLLMClient(
            "http://localhost:8788/v1",
            model="webqsp-agent",
            api_key="token",
            extra_body={"enable_thinking": False},
        )

        with patch("kgqa.serving.llm.client.requests.post", return_value=FakeOpenAIResponse()) as post:
            client.generate("hello", use_adapter=False)

        _args, kwargs = post.call_args
        self.assertEqual(kwargs["json"]["enable_thinking"], False)

    def test_openai_compatible_client_sleeps_and_retries_429(self):
        client = OpenAICompatibleLLMClient(
            "http://localhost:8788/v1",
            model="webqsp-agent",
            api_key="token",
        )

        with patch(
            "kgqa.serving.llm.client.requests.post",
            side_effect=[FakeRateLimitResponse(), FakeOpenAIResponse()],
        ) as post, patch("kgqa.serving.llm.client.time.sleep") as sleep:
            response = client.generate("hello", use_adapter=False)

        self.assertEqual(response.text, "ok")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(60.0)

    def test_siliconflow_client_defaults_to_qwen_non_thinking(self):
        with patch.dict("os.environ", {"SILICONFLOW_API_KEY": "sf-token"}, clear=True):
            client = SiliconFlowLLMClient(timeout=9)

        with patch("kgqa.serving.llm.client.requests.post", return_value=FakeOpenAIResponse()) as post:
            response = client.generate(
                "hello",
                use_adapter=True,
                max_new_tokens=8,
                temperature=0.1,
                system_prompt="sys",
            )

        self.assertEqual(response.text, "ok")
        self.assertEqual(response.used_adapter, False)
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.siliconflow.cn/v1/chat/completions")
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer sf-token"})
        self.assertEqual(
            kwargs["json"],
            {
                "model": "zai-org/GLM-4.5-Air",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hello"},
                ],
                "max_tokens": 8,
                "temperature": 0.1,
                "enable_thinking": False,
            },
        )

    def test_siliconflow_client_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                SiliconFlowLLMClient()

    def test_generate_can_enable_adapter(self):
        client = LLMClient("http://localhost:8788")

        with patch("kgqa.serving.llm.client.requests.post", return_value=FakeResponse()) as post:
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

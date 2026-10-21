import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock


class MockHTTPTransport:
    """Capture the JSON body that a provider SDK would send over HTTP."""

    def __init__(self):
        self.payload = None

    def post(self, body):
        self.payload = json.loads(json.dumps(body))
        return {"choices": [{"message": {"content": "ok"}}]}


transport = MockHTTPTransport()
fake_litellm = types.ModuleType("litellm")


def completion(**kwargs):
    body = {
        key: kwargs[key]
        for key in ("model", "messages", "temperature")
        if key in kwargs
    }
    response = transport.post(body)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=response["choices"][0]["message"]["content"]))]
    )


fake_litellm.completion = completion
fake_litellm.completion_cost = Mock(return_value=0)
fake_litellm.success_callback = []
fake_litellm.failure_callback = []
sys.modules["litellm"] = fake_litellm

fake_dotenv = types.ModuleType("dotenv")
fake_dotenv.load_dotenv = Mock()
sys.modules["dotenv"] = fake_dotenv

from mllm_tools.litellm import LiteLLMWrapper  # noqa: E402


class FakeGeminiWrapper:
    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature


fake_moviepy = types.ModuleType("moviepy")
fake_moviepy.VideoFileClip = object
sys.modules["moviepy"] = fake_moviepy

fake_gemini = types.ModuleType("mllm_tools.gemini")
fake_gemini.GeminiWrapper = FakeGeminiWrapper
sys.modules["mllm_tools.gemini"] = fake_gemini

fake_eval_utils = types.ModuleType("eval_suite.utils")
fake_eval_utils.calculate_geometric_mean = Mock()
sys.modules["eval_suite.utils"] = fake_eval_utils

fake_text_utils = types.ModuleType("eval_suite.text_utils")
fake_text_utils.parse_srt_to_text = Mock()
fake_text_utils.fix_transcript = Mock()
fake_text_utils.evaluate_text = Mock()
sys.modules["eval_suite.text_utils"] = fake_text_utils

fake_video_utils = types.ModuleType("eval_suite.video_utils")
fake_video_utils.evaluate_video_chunk_new = Mock()
sys.modules["eval_suite.video_utils"] = fake_video_utils

fake_image_utils = types.ModuleType("eval_suite.image_utils")
fake_image_utils.evaluate_sampled_images = Mock()
sys.modules["eval_suite.image_utils"] = fake_image_utils

import evaluate  # noqa: E402


class MiniMaxTextBoundaryTest(unittest.TestCase):
    def setUp(self):
        transport.payload = None

    def test_catalog_orders_flash_before_m3(self):
        self.assertEqual(
            evaluate.ALLOWED_MODELS[-2:],
            ["minimax/MiniMax-M3.1-Flash", "minimax/MiniMax-M3"],
        )
        self.assertEqual(evaluate.DEFAULT_TEXT_MODEL, "minimax/MiniMax-M3.1-Flash")

    def test_flash_text_payload_reaches_mock_http_boundary(self):
        model = LiteLLMWrapper(
            model_name="minimax/MiniMax-M3.1-Flash",
            temperature=0.0,
            use_langfuse=False,
        )

        self.assertEqual(model([{"type": "text", "content": "hello"}]), "ok")
        self.assertEqual(
            transport.payload,
            {
                "model": "minimax/MiniMax-M3.1-Flash",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ],
                "temperature": 0.0,
            },
        )

    def test_m3_text_payload_is_available_without_changing_non_minimax_shape(self):
        model = LiteLLMWrapper(
            model_name="minimax/MiniMax-M3",
            temperature=0.0,
            use_langfuse=False,
        )

        self.assertEqual(model([{"type": "text", "content": "m3"}]), "ok")
        self.assertEqual(transport.payload["model"], "minimax/MiniMax-M3")
        self.assertEqual(transport.payload["messages"][0]["content"][0]["type"], "text")

        model = LiteLLMWrapper(
            model_name="gpt-4o",
            temperature=0.0,
            use_langfuse=False,
        )
        self.assertEqual(model([{"type": "text", "content": "gpt"}]), "ok")
        self.assertEqual(transport.payload["model"], "gpt-4o")
        self.assertEqual(transport.payload["messages"][0]["content"][0]["text"], "gpt")

    def test_explicit_legacy_minimax_text_request_remains_accepted(self):
        model = LiteLLMWrapper(
            model_name="minimax/MiniMax-M2.7",
            temperature=0.0,
            use_langfuse=False,
        )

        self.assertEqual(model([{"type": "text", "content": "legacy"}]), "ok")
        self.assertEqual(transport.payload["model"], "minimax/MiniMax-M2.7")


if __name__ == "__main__":
    unittest.main()

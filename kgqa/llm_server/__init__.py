"""兼容路径：LLM 服务已迁至 :mod:`kgqa.serving.llm`。"""
from kgqa.serving.llm import GenerateResponse, LLMClient, OpenAICompatibleLLMClient, SiliconFlowLLMClient

__all__ = ["GenerateResponse", "LLMClient", "OpenAICompatibleLLMClient", "SiliconFlowLLMClient"]

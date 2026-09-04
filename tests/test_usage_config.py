from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import mini_agent.llm.model as model_module
from mini_agent.llm import LLMResponse, LLMUsage, ModelConfig
from mini_agent.runtime_config import RuntimeConfig


def test_usage_configs_are_frozen_and_validate_positive_limits():
    model = ModelConfig(model="test-model")
    runtime = RuntimeConfig()
    assert (model.chat_max_output_tokens, model.compaction_max_output_tokens,
            model.max_attempts) == (2048, 4096, 3)
    assert (runtime.max_steps, runtime.max_chat_usage,
            runtime.max_compaction_usage) == (8, 50_000, 800_000)
    with pytest.raises(ValidationError):
        RuntimeConfig(max_chat_usage=0)
    with pytest.raises(ValidationError):
        RuntimeConfig(max_steps=0)
    with pytest.raises(ValidationError):
        model.max_attempts = 4


async def test_client_selects_output_limit_and_returns_usage(monkeypatch):
    requests = []

    class Completions:
        async def create(self, **request):
            requests.append(request)
            message = SimpleNamespace(model_dump=lambda **_: {"content": "Answer"})
            usage = SimpleNamespace(prompt_tokens=90, completion_tokens=10,
                                    total_tokens=100)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                usage=usage,
            )

    class SDK:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("API_KEY", "key")
    monkeypatch.setenv("BASE_URL", "https://example.test")
    monkeypatch.setattr(model_module, "AsyncOpenAI", SDK)
    client = model_module.DeepSeekClient(ModelConfig(
        model="test-model", chat_max_output_tokens=100,
        compaction_max_output_tokens=200,
    ))
    chat = await client.llm_call(purpose="chat", messages=[], tools=[])
    compact = await client.llm_call(purpose="compaction", messages=[], tools=[])
    assert [request["max_tokens"] for request in requests] == [100, 200]
    assert chat == LLMResponse(
        message={"content": "Answer"},
        usage=LLMUsage(prompt_tokens=90, completion_tokens=10, total_tokens=100),
        finish_reason="stop",
    )
    assert compact.usage == chat.usage

from types import SimpleNamespace

import mini_agent.llm.model as model_module


async def test_deepseek_model_loads_env_and_uses_llm_call(monkeypatch) -> None:
    captured: dict = {}

    class FakeCompletions:
        async def create(self, **request):
            captured["request"] = request
            message = SimpleNamespace(
                model_dump=lambda **kwargs: {
                    "role": "assistant",
                    "content": "answer",
                }
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key: str, base_url: str, max_retries: int) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["max_retries"] = max_retries
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "https://example.test")
    monkeypatch.setenv("MODEL", "test-model")
    monkeypatch.setattr(model_module, "AsyncOpenAI", FakeAsyncOpenAI)

    client = model_module.DeepSeekClient()
    response = await client.llm_call(
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )

    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://example.test"
    assert captured["max_retries"] == 0
    assert captured["request"]["model"] == "test-model"
    assert "tools" not in captured["request"]
    assert response == {"role": "assistant", "content": "answer"}

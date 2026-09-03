import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

import mini_agent.llm.model as model_module
from mini_agent import AgentDefinition, ContextPolicy


def response(content="Answer"):
    message = SimpleNamespace(model_dump=lambda **kwargs: {"role": "assistant", "content": content})
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def request_error(kind):
    request = httpx.Request("POST", "https://example.test/chat/completions")
    if kind == "connection":
        return APIConnectionError(request=request)
    if kind == "timeout":
        return APITimeoutError(request=request)
    return APIStatusError(
        "Request failed", response=httpx.Response(kind, request=request), body=None,
    )


@pytest.fixture
def make_client(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "https://example.test")
    monkeypatch.setenv("MODEL", "test-model")
    monkeypatch.setattr(model_module, "load_dotenv", lambda: None)

    def build(outcomes):
        create = AsyncMock(side_effect=outcomes)
        sdk = Mock(return_value=SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )))
        sleep = AsyncMock()
        monkeypatch.setattr(model_module, "AsyncOpenAI", sdk)
        monkeypatch.setattr(model_module.asyncio, "sleep", sleep)
        client = model_module.DeepSeekClient()
        sdk.assert_called_once_with(
            api_key="test-key", base_url="https://example.test", max_retries=0,
        )
        return client, create, sleep

    return build


@pytest.mark.parametrize("kind", ["connection", "timeout", 429, 500, 502, 503, 504])
async def test_transient_failure_retries_same_request_without_mutating_inputs(make_client, kind):
    client, create, sleep = make_client([request_error(kind), response()])
    messages = [{"role": "user", "content": "Hi"}]
    tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
    original = deepcopy((messages, tools))
    result = await client.llm_call(messages=messages, tools=tools)
    assert result["content"] == "Answer"
    assert create.await_count == 2
    assert create.await_args_list[0] == create.await_args_list[1]
    sleep.assert_awaited_once_with(1)
    assert (messages, tools) == original


async def test_success_on_third_attempt_uses_one_then_two_second_delays(make_client):
    client, create, sleep = make_client([request_error(429), request_error(503), response()])
    await client.llm_call(messages=[], tools=[])
    assert create.await_count == 3
    assert sleep.await_args_list == [call(1), call(2)]


async def test_exhaustion_reraises_last_error_without_fourth_attempt_or_final_sleep(make_client):
    errors = [request_error(503) for _ in range(3)]
    client, create, sleep = make_client(errors)
    with pytest.raises(APIStatusError) as raised:
        await client.llm_call(messages=[], tools=[])
    assert raised.value is errors[-1]
    assert create.await_count == 3
    assert sleep.await_args_list == [call(1), call(2)]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 409, 422])
async def test_nonretryable_http_errors_fail_immediately(make_client, status):
    error = request_error(status)
    client, create, sleep = make_client([error])
    with pytest.raises(APIStatusError) as raised:
        await client.llm_call(messages=[], tools=[])
    assert raised.value is error
    assert create.await_count == 1
    sleep.assert_not_awaited()


@pytest.mark.parametrize("error", [RuntimeError("bug"), asyncio.CancelledError()])
async def test_unexpected_errors_and_cancellation_are_not_retried(make_client, error):
    client, create, sleep = make_client([error])
    with pytest.raises(type(error)):
        await client.llm_call(messages=[], tools=[])
    assert create.await_count == 1
    sleep.assert_not_awaited()


async def test_invalid_successful_response_is_not_retried(make_client):
    client, create, sleep = make_client([SimpleNamespace(choices=[])])
    with pytest.raises(IndexError):
        await client.llm_call(messages=[], tools=[])
    assert create.await_count == 1
    sleep.assert_not_awaited()


async def test_retry_does_not_increase_agent_steps_or_duplicate_messages(make_client):
    client, create, sleep = make_client([request_error(503), response()])
    runtime = AgentDefinition("Answer clearly", []).create_runtime(llm_client=client, max_steps=1)
    result = await runtime.run("Hi")
    assert result.status == "completed"
    assert result.steps_used == 1
    assert len(result.new_messages) == 2
    assert create.await_count == 2


async def test_summary_request_also_retries_through_shared_client(make_client):
    client, create, sleep = make_client([
        request_error("timeout"), response("Summary"), response("Answer"),
    ])
    runtime = AgentDefinition("Answer clearly", []).create_runtime(
        llm_client=client, max_steps=1,
        context_policy=ContextPolicy(context_limit=2500, trigger_ratio=0.6,
                                     max_summary_chars=100, output_reserve=100),
    )
    history = [{"role": "user", "content": "x" * 6000},
               {"role": "assistant", "content": "Old answer"}]
    for index in range(4):
        history.extend([{"role": "user", "content": str(index)},
                        {"role": "assistant", "content": "Recent"}])
    result = await runtime.run("Follow up", history)
    assert result.status == "completed" and result.compacted
    assert result.steps_used == 1
    assert create.await_count == 3
    assert create.await_args_list[0] == create.await_args_list[1]
    assert "Summarize" in create.await_args_list[0].kwargs["messages"][0]["content"]
    sleep.assert_awaited_once_with(1)

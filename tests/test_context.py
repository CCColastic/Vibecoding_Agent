from copy import deepcopy
import json

import pytest

from mini_agent.context import (
    CompactionError, ContextCompactor, ContextLimitExceeded, ContextPolicy,
)
from mini_agent.context.compactor import estimate_tokens, request_messages
from tests.fakes import FailingLLMClient, FakeLLMClient, tool_call


def turn(text):
    return [{"role": "user", "content": text}, {"role": "assistant", "content": "OK"}]


def history():
    return turn("上海 mock weather " + "旧" * 5600) + [
        message for index in range(5) for message in turn(f"Recent {index}")
    ]


def policy(**overrides):
    return ContextPolicy(**{
        "context_limit": 2000, "output_reserve": 200, "max_summary_chars": 120,
        **overrides,
    })


async def prepare(client, messages, *, current=None, config=None):
    current = current if current is not None else [{"role": "user", "content": "Follow up"}]
    return await ContextCompactor(llm_client=client, policy=config or policy()).prepare(
        messages=[*messages, *current], current_messages=current,
        system_prompt="Remember facts", tools=[],
    )


@pytest.mark.parametrize("overrides", [
    {"context_limit": 0}, {"output_reserve": -1}, {"output_reserve": 2000},
    {"trigger_ratio": 0}, {"trigger_ratio": 1}, {"trigger_ratio": float("nan")},
    {"keep_recent_turns": 0}, {"max_summary_chars": 0},
])
def test_policy_rejects_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        policy(**overrides)


def test_estimate_counts_unicode_messages_and_tools_without_ascii_escaping():
    messages = [{"role": "user", "content": "上海天气"}]
    tools = [{"name": "weather", "description": "查询天气"}]
    expected = len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)) / 4
    assert estimate_tokens(messages, tools) == expected
    assert estimate_tokens(messages, tools) > estimate_tokens(messages, [])


async def test_below_threshold_does_not_call_llm_and_strips_only_internal_metadata():
    source = [{"role": "assistant", "content": "Historical fact", "_kind": "context_summary"},
              *turn("Hello")]
    original = deepcopy(source)
    client = FakeLLMClient([])
    result = await prepare(client, source)
    assert not result.compacted
    assert client.calls == []
    wire = request_messages("System", result.messages)
    assert wire[0] == {"role": "system", "content": "System"}
    assert all("_kind" not in message for message in wire)
    assert source == original
    assert result.messages[0]["_kind"] == "context_summary"


async def test_compacts_at_exact_threshold_and_preserves_four_turns_and_current(caplog):
    source = history()
    original = deepcopy(source)
    current = [{"role": "user", "content": "Follow up"}]
    before = estimate_tokens(request_messages("Remember facts", [*source, *current]), [])
    config = policy(context_limit=int(before * 2), trigger_ratio=0.5)
    assert config.context_limit * config.trigger_ratio <= before
    client = FakeLLMClient([{"content": "Shanghai weather was mock, not real."}])
    with caplog.at_level("INFO"):
        result = await prepare(client, source, current=current, config=config)
    assert result.compacted
    assert result.messages[0]["_kind"] == "context_summary"
    assert result.messages[1:] == [*source[-8:], *current]
    assert source == original
    assert client.calls[0]["tools"] == []
    supplied = json.loads(client.calls[0]["messages"][1]["content"])
    assert supplied == source[:-8]
    assert "mock" in client.calls[0]["messages"][0]["content"]
    assert "context_compaction_success" in caplog.text
    assert "旧旧旧" not in caplog.text


async def test_rolling_summary_merges_previous_summary_exactly_once():
    previous = {"role": "assistant", "content": "Previous fact", "_kind": "context_summary"}
    source = [previous, *history()]
    client = FakeLLMClient([{"content": "Merged facts"}])
    result = await prepare(client, source)
    assert sum(m.get("_kind") == "context_summary" for m in result.messages) == 1
    supplied = json.loads(client.calls[0]["messages"][1]["content"])
    assert supplied == [{"role": "assistant", "content": "Previous fact"}, *source[1:-8]]
    assert "Previous fact" not in result.messages[0]["content"]


async def test_never_reduces_configured_retention_to_fit_budget():
    source = turn("x" * 3500) + [m for _ in range(4) for m in turn("y" * 600)]
    original = deepcopy(source)
    client = FakeLLMClient([])
    with pytest.raises(ContextLimitExceeded, match="Protected turns"):
        await prepare(client, source, config=policy(trigger_ratio=0.35))
    assert source == original
    assert client.calls == []


@pytest.mark.parametrize("turn_count", [2, 3, 4])
async def test_at_or_below_retention_count_keeps_all_history(turn_count):
    source = turn("x" * 5600) + [m for _ in range(turn_count - 1) for m in turn("Recent")]
    original = deepcopy(source)
    client = FakeLLMClient([])
    with pytest.raises(ContextLimitExceeded, match="No older turns"):
        await prepare(client, source)
    assert source == original
    assert client.calls == []


async def test_summary_limit_is_not_reserved_against_retained_turns():
    source = history()
    client = FakeLLMClient([{"content": "Short summary"}])
    # The maximum allowed summary would not fit, but the actual summary does.
    result = await prepare(client, source, config=policy(max_summary_chars=8000))
    assert result.compacted
    assert result.messages[1:-1] == source[-8:]
    assert json.loads(client.calls[0]["messages"][1]["content"]) == source[:-8]


async def test_retained_turns_are_not_limited_by_max_summary_chars():
    source = turn("x" * 5600) + [m for _ in range(4) for m in turn("recent" * 50)]
    client = FakeLLMClient([{"content": "Short summary"}])
    result = await prepare(client, source)
    assert result.messages[1:-1] == source[-8:]
    assert sum(len(m["content"]) for m in result.messages[1:-1]) > policy().max_summary_chars


async def test_retained_tool_call_and_result_are_not_split():
    tool_turn = [
        {"role": "user", "content": "Search"},
        {"role": "assistant", "content": None,
         "tool_calls": [tool_call("search", '{"query":"x"}')]},
        {"role": "tool", "tool_call_id": "call-1", "content": "mock result"},
        {"role": "assistant", "content": "Found it"},
    ]
    result = await prepare(FakeLLMClient([{"content": "Summary"}]),
                           [*history(), *tool_turn])
    assert result.messages[-5:-1] == tool_turn


@pytest.mark.parametrize("response", [
    None, {"content": ""}, {"content": "   "}, {"content": None},
    {"content": "x" * 121}, {"content": "Summary", "tool_calls": [tool_call("search", "{}")]},
])
async def test_invalid_summary_never_mutates_source_or_retries(response):
    source = history()
    original = deepcopy(source)
    client = FakeLLMClient([response])
    with pytest.raises(CompactionError):
        await prepare(client, source)
    assert source == original
    assert len(client.calls) == 1


async def test_summary_exception_is_safe_and_does_not_leak_payload(caplog):
    with pytest.raises(CompactionError, match="Summary request failed: RuntimeError") as error:
        await prepare(FailingLLMClient(), history())
    assert "secret" not in str(error.value)
    assert "secret" not in caplog.text


async def test_summary_that_expands_serialized_context_is_rejected():
    # Control characters expand in JSON even though the text character count fits.
    client = FakeLLMClient([{"content": "\x01" * 1600}])
    config = policy(context_limit=3000, max_summary_chars=1600, keep_recent_turns=1)
    with pytest.raises(CompactionError, match="did not reduce"):
        await prepare(client, [*turn("x" * 8500), *turn("recent")], config=config)


@pytest.mark.parametrize("source,current", [
    ([], [{"role": "user", "content": "x" * 8000}]),
    (turn("x" * 8000), [{"role": "user", "content": "Follow up"}]),
    (history(), [{"role": "user", "content": "x" * 8000}]),
])
async def test_uncompressible_protected_context_stops_before_llm(source, current):
    client = FakeLLMClient([])
    with pytest.raises(ContextLimitExceeded):
        await prepare(client, source, current=current)
    assert client.calls == []


async def test_summary_request_too_large_stops_without_chunking():
    client = FakeLLMClient([])
    with pytest.raises(ContextLimitExceeded, match="Summary request"):
        await prepare(client, [*turn("x" * 20000), *turn("Recent")],
                      config=policy(keep_recent_turns=1))
    assert client.calls == []

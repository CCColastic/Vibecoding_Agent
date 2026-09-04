from copy import deepcopy
import json

import pytest

from mini_agent.context import (
    CompactionError, ContextCompactor, ContextLimitExceeded, ContextPolicy,
)
from mini_agent.context.compactor import estimate_tokens, request_messages
from mini_agent.llm.run import RunLLM, RunUsage
from mini_agent.runtime_config import RuntimeConfig
from tests.fakes import FakeLLMClient, tool_call


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
    return await ContextCompactor(policy=config or policy()).prepare(
        messages=[*messages, *current], current_messages=current,
        system_prompt="Remember facts", tools=[],
        run_llm=RunLLM(client=client, config=RuntimeConfig(), usage=RunUsage()),
    )


def test_estimate_counts_unicode_messages_and_tools_without_ascii_escaping():
    messages = [{"role": "user", "content": "上海天气"}]
    tools = [{"name": "weather", "description": "查询天气"}]
    expected = len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)) / 4
    assert estimate_tokens(messages, tools) == expected
    assert estimate_tokens(messages, tools) > estimate_tokens(messages, [])


async def test_compacts_at_exact_threshold_and_preserves_four_turns_and_current(caplog):
    tool_turn = [
        {"role": "user", "content": "Search"},
        {"role": "assistant", "content": None,
         "tool_calls": [tool_call("search", '{"query":"x"}')]},
        {"role": "tool", "tool_call_id": "call-1", "content": "mock result"},
        {"role": "assistant", "content": "Found it"},
    ]
    source = history()[:-2] + tool_turn
    current = [{"role": "user", "content": "Follow up"}]
    before = estimate_tokens(request_messages("Remember facts", [*source, *current]), [])
    if before * 2 != int(before * 2):
        source[0]["content"] += "x"
        before = estimate_tokens(request_messages("Remember facts", [*source, *current]), [])
    original = deepcopy(source)
    config = policy(context_limit=int(before * 2), trigger_ratio=0.5)
    assert config.context_limit * config.trigger_ratio == before
    client = FakeLLMClient([{"content": "Shanghai weather was mock, not real."}])
    with caplog.at_level("INFO"):
        result = await prepare(client, source, current=current, config=config)
    assert result.compacted
    assert result.messages[0]["_kind"] == "context_summary"
    assert result.messages[1:] == [*source[4:], *current]
    assert result.messages[-5:-1] == tool_turn
    assert source == original
    assert client.calls[0]["tools"] == []
    supplied = json.loads(client.calls[0]["messages"][1]["content"])
    assert supplied == source[:4]
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


@pytest.mark.parametrize("response", [
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

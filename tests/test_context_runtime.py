import json
from copy import deepcopy

from mini_agent import AgentDefinition, ContextPolicy, ToolResult
from mini_agent.context.compactor import estimate_tokens, request_messages
from mini_agent.tools import SearchTool
from tests.fakes import FakeLLMClient, message_payloads, tool_call


def long_history():
    return [
        {"role": "user", "content": "旧" * 5600},
        {"role": "assistant", "content": "Old answer"},
        {"role": "user", "content": "Recent"},
        {"role": "assistant", "content": "Recent answer"},
    ]


def small_policy(**overrides):
    return ContextPolicy(**{
        "context_limit": 2000, "output_reserve": 200, "max_summary_chars": 120,
        "keep_recent_turns": 1,
        **overrides,
    })


async def test_runtime_compaction_separates_effective_history_and_new_messages():
    source = long_history()
    original = deepcopy(source)
    client = FakeLLMClient([
        {"content": "Summary of old facts"}, {"content": "Final answer"},
        {"content": "Follow up answer"},
    ])
    runtime = AgentDefinition("Remember", []).create_runtime(
        llm_client=client, context_policy=small_policy(), max_steps=1,
    )
    compactor = runtime.compactor
    result = await runtime.run("Question", source)
    assert result.status == "completed"
    assert result.steps_used == 1
    assert len(client.calls) == 2
    assert result.compacted
    assert message_payloads(result.new_messages) == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Final answer"},
    ]
    assert result.messages[1:] == [*source[-2:], *result.new_messages]
    assert all(m["role"] != "system" for m in result.messages)
    assert all("_kind" not in m for m in client.calls[1]["messages"])
    assert source == original
    follow_up = await runtime.run("Again", result.messages)
    assert follow_up.steps_used == 1
    assert not follow_up.compacted
    assert runtime.compactor is compactor
    assert compactor.llm_client is runtime.llm_client is client
    assert client.calls[-1]["messages"][1]["content"] == result.messages[0]["content"]


async def test_tool_growth_triggers_check_before_next_llm_call(monkeypatch):
    async def large_result(self, *, query):
        return ToolResult(ok=True, content="result " * 180)

    monkeypatch.setattr(SearchTool, "execute", large_result)
    definition = AgentDefinition("Remember", [SearchTool()])
    source = long_history()
    source[0]["content"] = "old " * 900
    current = {"role": "user", "content": "Search now"}
    # Set the trigger just above the initial request but below the tool response.
    probe = definition.create_runtime(llm_client=FakeLLMClient([]))
    before = estimate_tokens(request_messages("Remember", [*source, current]), probe.registry.schemas())
    config = small_policy(context_limit=int(before / 0.7) + 100, output_reserve=100)
    call = {"role": "assistant", "content": None,
            "tool_calls": [tool_call("search", '{"query":"hello"}')]}
    responses = [call, {"content": "Summary"}, {"content": "Done"}]
    client = FakeLLMClient(responses)
    runtime = definition.create_runtime(llm_client=client, context_policy=config, max_steps=2)
    result = await runtime.run(current["content"], source)
    assert result.steps_used == 2
    assert result.status == "completed"
    assert len(result.tool_executions) == 1
    assert result.new_messages[0] == {**current, "_run_id": result.run_id}
    assert result.new_messages[1] == {**call, "_run_id": result.run_id}
    assert result.new_messages[2]["tool_call_id"] == "call-1"
    assert client.calls[1]["tools"] == []
    summary_input = json.loads(client.calls[1]["messages"][1]["content"])
    assert summary_input == source[:-2]
    assert result.compacted
    assert client.calls[2]["messages"][-3:] == message_payloads(result.new_messages[:3])
    assert result.new_messages[-1]["content"] == "Done"
    assert len(result.new_messages) == 4

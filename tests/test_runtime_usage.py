import json

from mini_agent import AgentDefinition
from mini_agent import ContextPolicy
from mini_agent.llm import ModelConfig
from mini_agent.runtime_config import RuntimeConfig
from mini_agent.tools import CalculatorTool
from mini_agent.trace import TraceRecorder
from tests.fakes import FakeLLMClient, llm_response, tool_call


def traced_runtime(tmp_path, responses, *, chat_limit=50_000):
    client = FakeLLMClient(responses)
    client.config = ModelConfig(model="fake")
    runtime = AgentDefinition("Use tools", [CalculatorTool()]).create_runtime(
        llm_client=client,
        runtime_config=RuntimeConfig(max_steps=3, max_chat_usage=chat_limit),
        trace_recorder=TraceRecorder(tmp_path),
    )
    return runtime, client


async def test_runtime_accumulates_usage_in_result_and_every_trace_event(tmp_path):
    call = tool_call("calculator", '{"operation":"add","a":1,"b":2}')
    runtime, _ = traced_runtime(tmp_path, [
        llm_response({"content": None, "tool_calls": [call]},
                     prompt_tokens=90, completion_tokens=10),
        llm_response({"content": "3"}, prompt_tokens=180, completion_tokens=20),
    ])
    result = await runtime.run("Calculate")
    assert (result.chat_token_usage, result.compaction_token_usage,
            result.token_usage, result.usage_complete) == (300, 0, 300, True)
    trace = json.loads((tmp_path / f"{result.run_id}.json").read_text("utf-8"))
    assert [event["token_usage"] for event in trace] == [0, 100, 100, 100, 300, 300]
    assert all("chat_token_usage" in event and "compaction_token_usage" in event
               and "usage_complete" in event for event in trace)
    assert trace[1]["data"]["usage"] == {
        "prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100,
    }
    assert trace[-1]["token_usage"] == result.token_usage


async def test_budget_exhausted_by_tool_response_stops_before_tool_execution(tmp_path):
    call = tool_call("calculator", '{"operation":"add","a":1,"b":2}')
    runtime, client = traced_runtime(tmp_path, [
        llm_response({"content": None, "tool_calls": [call]},
                     prompt_tokens=90, completion_tokens=10),
    ], chat_limit=100)
    result = await runtime.run("Calculate")
    assert result.status == "token_budget_exceeded"
    assert result.token_usage == 100
    assert result.tool_executions == []
    assert len(result.new_messages) == 1
    assert len(client.calls) == 1


async def test_complete_final_answer_can_soft_exceed_budget(tmp_path):
    runtime, _ = traced_runtime(tmp_path, [
        llm_response({"content": "answer"}, prompt_tokens=100, completion_tokens=1),
    ], chat_limit=100)
    result = await runtime.run("Question")
    assert result.status == "completed"
    assert result.final_answer == "answer"
    assert result.token_usage == 101


async def test_truncated_response_is_not_added_to_history_or_executed(tmp_path):
    call = tool_call("calculator", '{"operation":"add","a":1,"b":2}')
    runtime, _ = traced_runtime(tmp_path, [
        llm_response({"content": "partial", "tool_calls": [call]},
                     prompt_tokens=90, completion_tokens=10,
                     finish_reason="length"),
    ])
    result = await runtime.run("Calculate")
    assert result.status == "output_truncated"
    assert len(result.new_messages) == 1
    assert result.tool_executions == []


async def test_missing_usage_keeps_answer_and_marks_trace_incomplete(tmp_path):
    runtime, _ = traced_runtime(tmp_path, [
        llm_response({"content": "answer"}, usage_available=False),
    ])
    result = await runtime.run("Question")
    assert result.status == "completed"
    assert result.token_usage == 0
    assert not result.usage_complete
    trace = json.loads((tmp_path / f"{result.run_id}.json").read_text("utf-8"))
    assert trace[1]["data"]["usage"] is None
    assert trace[1]["data"]["usage_unavailable"] is True
    assert all(not event["usage_complete"] for event in trace[1:])


async def test_compaction_usage_has_its_own_bucket_and_trace_event(tmp_path):
    client = FakeLLMClient([
        llm_response({"content": "Summary"}, prompt_tokens=40, completion_tokens=10),
        llm_response({"content": "Answer"}, prompt_tokens=25, completion_tokens=5),
    ])
    client.config = ModelConfig(model="fake")
    runtime = AgentDefinition("Remember", []).create_runtime(
        llm_client=client,
        runtime_config=RuntimeConfig(max_chat_usage=100, max_compaction_usage=50),
        context_policy=ContextPolicy(
            context_limit=2_000, output_reserve=200,
            max_summary_chars=120, keep_recent_turns=1,
        ),
        trace_recorder=TraceRecorder(tmp_path),
    )
    history = [
        {"role": "user", "content": "旧" * 5_600},
        {"role": "assistant", "content": "Old"},
        {"role": "user", "content": "Recent"},
        {"role": "assistant", "content": "Keep"},
    ]
    result = await runtime.run("Continue", history)
    assert result.status == "completed" and result.compacted
    assert (result.chat_token_usage, result.compaction_token_usage,
            result.token_usage) == (30, 50, 80)
    trace = json.loads((tmp_path / f"{result.run_id}.json").read_text("utf-8"))
    assert [event["event"] for event in trace] == [
        "user.input", "context.compacted", "assistant.output", "run.end",
    ]
    assert [event["token_usage"] for event in trace] == [0, 50, 80, 80]
    assert trace[1]["data"]["purpose"] == "compaction"
    assert [call["purpose"] for call in client.calls] == ["compaction", "chat"]

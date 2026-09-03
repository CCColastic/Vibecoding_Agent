import asyncio
import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from mini_agent import AgentDefinition, ContextPolicy, ToolResult, TraceEvent, TraceRecorder
from mini_agent.tools import CalculatorTool
from tests.fakes import FakeLLMClient, tool_call


def events(directory, run_id):
    return json.loads((directory / f"{run_id}.json").read_text(encoding="utf-8"))


def runtime(directory, responses, **kwargs):
    return AgentDefinition("System secret should not be logged", [CalculatorTool()]).create_runtime(
        llm_client=FakeLLMClient(responses), trace_recorder=TraceRecorder(directory), **kwargs,
    )


async def test_trace_records_inputs_outputs_tools_and_one_end(tmp_path):
    call = tool_call("calculator", '{"operation":"multiply","a":6,"b":7}')
    agent = runtime(tmp_path, [
        {"content": None, "tool_calls": [call], "reasoning_content": "private reasoning"},
        {"content": "结果是 42"},
    ])
    result = await agent.run("计算 6 × 7", session_id="session-a")
    trace = events(tmp_path, result.run_id)
    assert [e["event"] for e in trace] == [
        "user.input", "assistant.output", "tool.start", "tool.end", "assistant.output", "run.end",
    ]
    assert [e["sequence"] for e in trace] == list(range(1, 7))
    assert [e["step"] for e in trace] == [0, 1, 1, 1, 2, 2]
    assert all(e["run_id"] == result.run_id and e["session_id"] == "session-a" for e in trace)
    assert all(datetime.fromisoformat(e["timestamp"]).utcoffset().total_seconds() == 0 for e in trace)
    assert trace[0]["data"] == {"content": "计算 6 × 7"}
    assert trace[1]["data"] == {"content": None, "tool_calls": [call]}
    assert trace[2]["data"]["raw_arguments"] == call["function"]["arguments"]
    assert trace[3]["data"]["tool_call_id"] == trace[2]["data"]["tool_call_id"]
    assert trace[3]["data"]["result"] == {"ok": True, "content": 42}
    assert trace[-1]["data"]["status"] == "completed"
    assert trace[-1]["data"]["steps_used"] == 2
    assert trace[-1]["data"]["duration_ms"] >= trace[3]["data"]["duration_ms"] >= 0
    assert "private reasoning" not in json.dumps(trace)
    assert "System secret" not in json.dumps(trace)
    content = (tmp_path / f"{result.run_id}.json").read_text(encoding="utf-8")
    assert content == json.dumps(trace, ensure_ascii=False, indent=2) + "\n"
    assert "结果是 42" in content
    assert not list(tmp_path.glob("*.jsonl"))


async def test_multiple_invalid_tool_calls_are_paired(tmp_path):
    calls = [tool_call("missing", "{}", call_id="one"),
             tool_call("calculator", "not-json", call_id="two")]
    agent = runtime(tmp_path, [{"content": None, "tool_calls": calls}, {"content": "Failed tools"}])
    result = await agent.run("Try tools")
    tool_events = [e for e in events(tmp_path, result.run_id) if e["event"].startswith("tool.")]
    assert [e["event"] for e in tool_events] == ["tool.start", "tool.end"] * 2
    assert [e["data"]["tool_call_id"] for e in tool_events] == ["one", "one", "two", "two"]
    assert all(not e["data"]["result"]["ok"] for e in tool_events if e["event"] == "tool.end")


@pytest.mark.parametrize("response,status", [
    (RuntimeError("provider secret"), "llm_error"),
    ({"content": None}, "llm_protocol_error"),
    ({"content": None, "tool_calls": [tool_call("missing", "{}")]}, "max_steps_exceeded"),
])
async def test_failure_has_exactly_one_run_end(tmp_path, response, status):
    result = await runtime(tmp_path, [response], max_steps=1).run("Question")
    trace = events(tmp_path, result.run_id)
    assert result.status == status
    assert sum(e["event"] == "run.end" for e in trace) == 1
    assert trace[-1]["data"]["status"] == status
    assert "provider secret" not in json.dumps(trace)


async def test_compaction_error_is_traced_without_summary_as_assistant_output(tmp_path):
    policy = ContextPolicy(context_limit=2000, output_reserve=200,
                           max_summary_chars=120, keep_recent_turns=1)
    agent = runtime(tmp_path, [{"content": ""}], context_policy=policy)
    history = [{"role": "user", "content": "旧" * 5600}, {"role": "assistant", "content": "Old"},
               {"role": "user", "content": "Recent"}, {"role": "assistant", "content": "OK"}]
    result = await agent.run("Follow up", history)
    trace = events(tmp_path, result.run_id)
    assert result.status == "compaction_error"
    assert [e["event"] for e in trace] == ["user.input", "run.end"]
    assert trace[-1]["data"]["steps_used"] == 0


@pytest.mark.parametrize("during_tool", [False, True])
@pytest.mark.parametrize("failure,status", [(asyncio.CancelledError, "cancelled"),
                                            (RuntimeError, "unhandled_error")])
async def test_cancel_and_unhandled_failure_propagate_with_one_end(tmp_path, monkeypatch, during_tool, failure, status):
    responses = [{"content": None, "tool_calls": [tool_call("calculator", "{}")] }] if during_tool else []
    agent = runtime(tmp_path, responses)
    run_id = str(uuid4())

    async def fail(*args, **kwargs):
        raise failure("internal secret")

    if during_tool:
        monkeypatch.setattr(agent.registry, "execute_tool_call", fail)
    else:
        monkeypatch.setattr(agent.compactor, "prepare", fail)
    with pytest.raises(failure):
        await agent.run("Question", run_id=run_id)
    trace = events(tmp_path, run_id)
    assert trace[-1]["data"]["status"] == status
    assert sum(e["event"] == "run.end" for e in trace) == 1
    assert "internal secret" not in json.dumps(trace)
    if during_tool:
        assert trace[-2]["event"] == "tool.end"
        assert not trace[-2]["data"]["result"]["ok"]


async def test_real_task_cancellation_flushes_end(tmp_path, monkeypatch):
    agent = runtime(tmp_path, [])
    started = asyncio.Event()

    async def wait_for_cancel(**kwargs):
        started.set()
        await asyncio.Future()

    monkeypatch.setattr(agent.llm_client, "llm_call", wait_for_cancel)
    run_id = str(uuid4())
    task = asyncio.create_task(agent.run("Wait", run_id=run_id))
    await started.wait()
    assert events(tmp_path, run_id)[0]["event"] == "user.input"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events(tmp_path, run_id)[-1]["data"]["status"] == "cancelled"


async def test_concurrent_runs_use_separate_files_and_sequences(tmp_path, monkeypatch):
    agent = runtime(tmp_path, [])

    async def respond(**kwargs):
        await asyncio.sleep(0)
        return {"content": "Answer"}

    monkeypatch.setattr(agent.llm_client, "llm_call", respond)
    results = await asyncio.gather(agent.run("A", session_id="a"), agent.run("B", session_id="b"))
    assert results[0].run_id != results[1].run_id
    for result, session in zip(results, ["a", "b"]):
        assert UUID(result.run_id).version == 4
        assert all(m["_run_id"] == result.run_id for m in result.new_messages)
        trace = events(tmp_path, result.run_id)
        assert [e["sequence"] for e in trace] == [1, 2, 3]
        assert all(e["session_id"] == session for e in trace)


async def test_trace_io_failure_does_not_change_result_or_expose_error(tmp_path, caplog):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("unchanged", encoding="utf-8")
    agent = runtime(blocked, [{"content": "Answer"}])
    result = await agent.run("Question")
    assert result.status == "completed"
    assert len(agent.llm_client.calls) == 1
    assert "Trace write failed" in caplog.text
    assert str(blocked) not in caplog.text


async def test_throwing_recorder_does_not_break_runtime(tmp_path, monkeypatch, caplog):
    agent = runtime(tmp_path, [{"content": "Answer"}])

    def fail(event):
        raise OSError("secret path")

    monkeypatch.setattr(agent.trace_recorder, "emit", fail)
    result = await agent.run("Question")
    assert result.status == "completed"
    assert "Trace recorder failed" in caplog.text
    assert "secret path" not in caplog.text


async def test_nonserializable_tool_result_uses_type_placeholder(tmp_path, monkeypatch):
    class PrivateValue:
        def __repr__(self):
            raise AssertionError("Must not use repr")

    async def execute(self, **kwargs):
        return ToolResult(ok=True, content=PrivateValue())

    monkeypatch.setattr(CalculatorTool, "execute", execute)
    call = tool_call("calculator", '{"operation":"add","a":1,"b":2}')
    result = await runtime(tmp_path, [{"tool_calls": [call]}, {"content": "Done"}]).run("Calculate")
    tool_end = next(e for e in events(tmp_path, result.run_id) if e["event"] == "tool.end")
    assert tool_end["data"]["result"]["content"] == {"unserializable_type": "PrivateValue"}


@pytest.mark.parametrize("run_id", ["../escape", "bad-id", str(UUID(int=0))])
async def test_invalid_run_id_never_creates_trace_file(tmp_path, run_id):
    agent = runtime(tmp_path / "traces", [])
    with pytest.raises(ValueError, match="UUID4"):
        await agent.run("Question", run_id=run_id)
    assert not (tmp_path / "traces").exists()


def test_recorder_rejects_unsafe_path_even_when_called_directly(tmp_path, caplog):
    TraceRecorder(tmp_path / "traces").emit(TraceEvent(
        datetime.now(timezone.utc), "../escape", None, 1, 0, "user.input", {"content": "secret"},
    ))
    assert not (tmp_path / "traces").exists()
    assert "secret" not in caplog.text


def test_failed_json_replacement_preserves_previous_events(tmp_path, monkeypatch, caplog):
    from pathlib import Path

    recorder = TraceRecorder(tmp_path)
    run_id = str(uuid4())
    first = TraceEvent(datetime.now(timezone.utc), run_id, None, 1, 0, "user.input", {"content": "Hello"})
    recorder.emit(first)
    original = (tmp_path / f"{run_id}.json").read_bytes()

    def fail_replace(self, target):
        raise OSError("private path")

    monkeypatch.setattr(Path, "replace", fail_replace)
    recorder.emit(TraceEvent(datetime.now(timezone.utc), run_id, None, 2, 1,
                             "assistant.output", {"content": "Answer"}))
    assert (tmp_path / f"{run_id}.json").read_bytes() == original
    assert len(events(tmp_path, run_id)) == 1
    assert not list(tmp_path.glob("*.tmp"))
    assert "Trace write failed" in caplog.text
    assert "private path" not in caplog.text


@pytest.mark.parametrize("content", ["broken JSON", '{"unexpected":"object"}'])
def test_invalid_existing_json_is_not_overwritten(tmp_path, content, caplog):
    run_id = str(uuid4())
    path = tmp_path / f"{run_id}.json"
    path.write_text(content, encoding="utf-8")
    TraceRecorder(tmp_path).emit(TraceEvent(datetime.now(timezone.utc), run_id, None,
                                          1, 0, "user.input", {"content": "Hello"}))
    assert path.read_text(encoding="utf-8") == content
    assert "Trace write failed" in caplog.text

import json

import pytest

from mini_agent import AgentDefinition, ContextPolicy
from mini_agent.session import ConversationManager, SQLiteSessionStore
from mini_agent.runtime_config import RuntimeConfig
from tests.fakes import FakeLLMClient, tool_call


class CountingSQLiteStore(SQLiteSessionStore):
    def __init__(self, path):
        self.loads = 0
        self.appends = 0
        self.replacements = 0
        self.fail_replace = False
        super().__init__(path)

    def load_messages(self, *args, **kwargs):
        self.loads += 1
        return super().load_messages(*args, **kwargs)

    def append_messages(self, *args, **kwargs):
        self.appends += 1
        return super().append_messages(*args, **kwargs)

    def replace_history(self, *args, **kwargs):
        self.replacements += 1
        if self.fail_replace:
            raise OSError("database unavailable")
        return super().replace_history(*args, **kwargs)


def setup_conversation(tmp_path, responses, *, max_steps=8):
    store = CountingSQLiteStore(tmp_path / "sessions.db")
    session = store.create_session("owner-a", "Long session")
    history = [
        {"role": "user", "content": "Old fact " + "旧" * 5600},
        {"role": "assistant", "content": "Old answer"},
        {"role": "user", "content": "Recent question"},
        {"role": "assistant", "content": "Recent answer"},
    ]
    SQLiteSessionStore.append_messages(store, "owner-a", session.id, "old", history)
    history = [{**message, "_run_id": "old"} for message in history]
    client = FakeLLMClient(responses)
    runtime = AgentDefinition("Remember", []).create_runtime(
        llm_client=client, runtime_config=RuntimeConfig(max_steps=max_steps),
        context_policy=ContextPolicy(
            context_limit=2000, max_summary_chars=120, output_reserve=200,
            keep_recent_turns=1,
        ),
    )
    manager = ConversationManager(owner_id="owner-a", runtime=runtime, store=store)
    return store, session, history, client, manager


def saved(store, session):
    return [{**m.payload, "_run_id": m.run_id}
            for m in SQLiteSessionStore.load_messages(store, "owner-a", session.id)]


async def test_compacted_session_replaces_once_then_appends_and_resumes_once(tmp_path):
    store, session, history, client, manager = setup_conversation(tmp_path, [
        {"content": "Old fact summary"}, {"content": "First answer"},
        {"content": "Second answer"}, {"content": "Resumed answer"},
        {"content": "Isolated answer"},
    ])
    conversation = manager.resume_conversation(session.id)
    first = await conversation.send_message("First question")
    assert first.compacted
    assert saved(store, session) == first.messages
    assert store.replacements == 1 and store.appends == 0 and store.loads == 1
    second = await conversation.send_message("Second question")
    assert saved(store, session) == [*first.messages, *second.new_messages]
    assert store.replacements == 1 and store.appends == 1 and store.loads == 1
    resumed = manager.resume_conversation(session.id)
    third = await resumed.send_message("Resume question")
    assert store.loads == 2
    assert saved(store, session) == third.messages
    assert client.calls[3]["messages"][1]["content"] == first.messages[0]["content"]
    assert all("_kind" not in m for m in client.calls[3]["messages"])
    other = manager.new_conversation()
    isolated = await other.send_message("Unrelated question")
    assert not isolated.compacted
    assert len(client.calls[-1]["messages"]) == 2
    assert saved(store, session) == third.messages


@pytest.mark.parametrize("summary", [{"content": ""}, RuntimeError("secret")])
async def test_summary_failure_skips_all_writes_and_retry_uses_original_memory(tmp_path, summary):
    store, session, history, client, manager = setup_conversation(tmp_path, [
        summary, {"content": "Summary"}, {"content": "Retry answer"},
    ])
    conversation = manager.resume_conversation(session.id)
    failed = await conversation.send_message("Do not persist me")
    assert failed.status == "compaction_error"
    assert failed.steps_used == 0
    assert not failed.compacted
    assert "secret" not in failed.error
    assert len(client.calls) == 1
    assert store.replacements == store.appends == 0
    assert saved(store, session) == history
    retried = await conversation.send_message("Retry")
    assert retried.status == "completed"
    assert store.loads == 1
    assert client.calls[0]["messages"] == client.calls[1]["messages"]
    assert "Do not persist me" not in json.dumps(retried.messages)
    assert saved(store, session) == retried.messages


async def test_replace_failure_leaves_memory_and_database_unchanged(tmp_path):
    store, session, history, client, manager = setup_conversation(tmp_path, [
        {"content": "First summary"}, {"content": "Unsaved answer"},
        {"content": "Retry summary"}, {"content": "Saved answer"},
    ])
    conversation = manager.resume_conversation(session.id)
    store.fail_replace = True
    with pytest.raises(OSError, match="database unavailable"):
        await conversation.send_message("Unsaved question")
    assert saved(store, session) == history
    store.fail_replace = False
    result = await conversation.send_message("Retry question")
    assert client.calls[0]["messages"] == client.calls[2]["messages"]
    assert store.loads == 1
    assert "Unsaved" not in json.dumps(result.messages)
    assert saved(store, session) == result.messages


async def test_successful_compaction_is_saved_after_llm_failure(tmp_path):
    store, session, history, client, manager = setup_conversation(
        tmp_path, [{"content": "Summary"}, RuntimeError("secret")], max_steps=1,
    )
    conversation = manager.resume_conversation(session.id)
    result = await conversation.send_message("Question")
    assert result.status == "llm_error"
    assert result.compacted
    assert store.replacements == 1 and store.appends == 0
    assert saved(store, session) == result.messages
    assert result.new_messages[0] == {
        "role": "user", "content": "Question", "_run_id": result.run_id,
    }


async def test_later_budget_failure_discards_even_an_earlier_successful_compaction(tmp_path):
    store, session, history, client, manager = setup_conversation(tmp_path, [
        {"content": "Summary"},
        {"content": "x" * 10000, "tool_calls": [tool_call("missing", "{}")]},
        {"content": "Retry summary"}, {"content": "Retry answer"},
    ])
    conversation = manager.resume_conversation(session.id)
    failed = await conversation.send_message("Unsaved question")
    assert failed.status == "context_limit_exceeded"
    assert failed.compacted
    assert failed.steps_used == 1
    assert len(failed.tool_executions) == 1
    assert len(failed.new_messages) == 3
    assert store.replacements == store.appends == 0
    assert saved(store, session) == history
    result = await conversation.send_message("Retry")
    assert result.status == "completed"
    assert client.calls[0]["messages"] == client.calls[2]["messages"]
    assert "Unsaved question" not in json.dumps(result.messages)
    assert saved(store, session) == result.messages

import json
import sqlite3
from uuid import uuid4

import pytest

from mini_agent import AgentDefinition, ContextPolicy
from mini_agent.app import build_conversation_manager
from mini_agent.cli import run_cli
from mini_agent.session import ConversationManager, SQLiteSessionStore
from tests.fakes import FakeLLMClient


async def test_compaction_and_resume_preserve_original_run_ids(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    session = store.create_session("owner", "History")
    old_id, recent_id = str(uuid4()), str(uuid4())
    store.append_messages("owner", session.id, old_id, [
        {"role": "user", "content": "旧" * 5600}, {"role": "assistant", "content": "Old"},
    ])
    store.append_messages("owner", session.id, recent_id, [
        {"role": "user", "content": "Recent"}, {"role": "assistant", "content": "Keep"},
    ])
    client = FakeLLMClient([{"content": "Summary"}, {"content": "Answer"}, {"content": "Again"}])
    agent = AgentDefinition("Remember", []).create_runtime(
        llm_client=client, context_policy=ContextPolicy(
            context_limit=2000, output_reserve=200, max_summary_chars=120, keep_recent_turns=1,
        ),
    )
    manager = ConversationManager(owner_id="owner", runtime=agent, store=store)
    result = await manager.resume_conversation(session.id).send_message("Question")
    assert result.compacted
    stored = store.load_messages("owner", session.id)
    assert [m.run_id for m in stored] == [result.run_id, recent_id, recent_id, result.run_id, result.run_id]
    assert all("_run_id" not in m.payload for m in stored)
    assert [{**m.payload, "_run_id": m.run_id} for m in stored] == result.messages
    summarized = json.loads(client.calls[0]["messages"][1]["content"])
    assert all("_run_id" not in m and "_kind" not in m for m in summarized)
    assert all("_run_id" not in m and "_kind" not in m for m in client.calls[1]["messages"])
    resumed = await manager.resume_conversation(session.id).send_message("Continue")
    assert resumed.run_id != result.run_id
    assert resumed.messages[:len(result.messages)] == result.messages
    assert {m.run_id for m in store.load_messages("owner", session.id)[-2:]} == {resumed.run_id}


def test_legacy_column_migration_is_idempotent_and_preserves_rows(tmp_path):
    database = tmp_path / "sessions.db"
    store = SQLiteSessionStore(database)
    session = store.create_session("owner", "Legacy")
    store.append_messages("owner", session.id, "legacy-id", [{"role": "user", "content": "Original"}])
    original = store.load_messages("owner", session.id)
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE messages RENAME COLUMN run_id TO turn_id")
    migrated = SQLiteSessionStore(database)
    assert migrated.load_messages("owner", session.id) == original
    assert SQLiteSessionStore(database).load_messages("owner", session.id) == original
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
    assert "run_id" in columns and "turn_id" not in columns


def test_replace_rejects_missing_identity_before_deleting(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    session = store.create_session("owner", "Keep")
    store.append_messages("owner", session.id, "original", [{"role": "user", "content": "Original"}])
    original = store.load_messages("owner", session.id)
    with pytest.raises(ValueError, match="_run_id"):
        store.replace_history("owner", session.id, [{"role": "user", "content": "Missing ID"}])
    assert store.load_messages("owner", session.id) == original


def test_append_rejects_mismatched_identity(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    session = store.create_session("owner", "Keep")
    with pytest.raises(ValueError, match="must match"):
        store.append_messages("owner", session.id, "one", [{"role": "user", "content": "Wrong", "_run_id": "two"}])
    assert store.load_messages("owner", session.id) == []


async def test_cli_no_trace_flag_and_run_id_output(tmp_path, monkeypatch):
    client = FakeLLMClient([{"content": "Answer"}])

    def build(**kwargs):
        assert kwargs == {"trace_enabled": False}
        return build_conversation_manager(data_dir=tmp_path, llm_client=client, **kwargs)

    monkeypatch.setattr("mini_agent.cli.build_conversation_manager", build)
    inputs = iter(["Question", "/exit"])
    output = []
    assert await run_cli(["new", "--no-trace"], input_fn=lambda _: next(inputs), output_fn=output.append) == 0
    assert any(line.startswith("Run: ") for line in output)
    assert not (tmp_path / "traces").exists()

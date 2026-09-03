import sqlite3

import pytest

from mini_agent.session import SQLiteSessionStore, SessionNotFoundError


def test_store_creates_appends_and_loads_session(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    session = store.create_session("owner-a", "First question")
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]

    updated = store.append_messages(
        "owner-a", session.id, "turn-1", messages
    )
    stored = store.load_messages("owner-a", session.id)

    assert updated.updated_at >= session.updated_at
    assert [message.sequence for message in stored] == [1, 2]
    assert [message.turn_id for message in stored] == ["turn-1", "turn-1"]
    assert [message.payload for message in stored] == messages

    store.append_messages(
        "owner-a",
        session.id,
        "turn-2",
        [{"role": "user", "content": "Follow-up"}],
    )
    assert [message.sequence for message in store.load_messages("owner-a", session.id)] == [
        1,
        2,
        3,
    ]


def test_store_enforces_owner_boundary(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    session = store.create_session("owner-a", "Private")

    with pytest.raises(SessionNotFoundError):
        store.get_session("owner-b", session.id)
    with pytest.raises(SessionNotFoundError):
        store.load_messages("owner-b", session.id)
    with pytest.raises(SessionNotFoundError):
        store.append_messages(
            "owner-b",
            session.id,
            "turn",
            [{"role": "user", "content": "No access"}],
        )


def test_store_lists_only_owner_sessions_by_recent_update(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    first = store.create_session("owner-a", "First")
    second = store.create_session("owner-a", "Second")
    store.create_session("owner-b", "Other owner")
    store.append_messages(
        "owner-a", first.id, "turn", [{"role": "user", "content": "new"}]
    )

    sessions = store.list_sessions("owner-a")

    assert [session.id for session in sessions] == [first.id, second.id]


def test_append_is_atomic_when_message_is_not_serializable(tmp_path) -> None:
    database_path = tmp_path / "sessions.db"
    store = SQLiteSessionStore(database_path)
    session = store.create_session("owner-a", "Atomic")

    with pytest.raises(TypeError):
        store.append_messages(
            "owner-a",
            session.id,
            "turn",
            [
                {"role": "user", "content": "first"},
                {"role": "tool", "content": object()},
            ],
        )

    with sqlite3.connect(database_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert count == 0


def test_replace_rebuilds_only_target_snapshot_and_append_continues(tmp_path):
    from uuid import UUID
    from tests.fakes import tool_call

    store = SQLiteSessionStore(tmp_path / "sessions.db")
    target = store.create_session("owner-a", "Target")
    other = store.create_session("owner-a", "Other")
    foreign = store.create_session("owner-b", "Private")
    for session in (target, other, foreign):
        store.append_messages(session.owner_id, session.id, "old-turn",
                              [{"role": "user", "content": "Old"}])
    snapshot = [
        {"role": "assistant", "content": "Summary", "_kind": "context_summary"},
        {"role": "user", "content": "Search"},
        {"role": "assistant", "content": None,
         "tool_calls": [tool_call("search", '{"query":"x"}')]},
        {"role": "tool", "tool_call_id": "call-1", "content": "mock"},
        {"role": "assistant", "content": "Done"},
        {"role": "user", "content": "Next"},
        {"role": "assistant", "content": "Answer"},
    ]
    updated = store.replace_history("owner-a", target.id, snapshot)
    saved = store.load_messages("owner-a", target.id)
    assert [m.payload for m in saved] == snapshot
    assert [m.sequence for m in saved] == list(range(1, 8))
    assert len({m.turn_id for m in saved[1:5]}) == 1
    assert saved[0].turn_id != saved[1].turn_id != saved[5].turn_id
    assert saved[5].turn_id == saved[6].turn_id
    assert all(UUID(m.turn_id).version == 4 for m in saved)
    assert all(m.created_at == updated.updated_at for m in saved)
    assert updated.created_at == target.created_at
    assert (updated.id, updated.title, updated.owner_id) == (target.id, target.title, target.owner_id)
    assert store.list_sessions("owner-a")[0].id == target.id
    for session in (other, foreign):
        assert store.load_messages(session.owner_id, session.id)[0].turn_id == "old-turn"
    store.append_messages("owner-a", target.id, "next-turn", [{"role": "user", "content": "Again"}])
    assert store.load_messages("owner-a", target.id)[-1].sequence == 8


def test_replace_rejects_foreign_owner_without_deleting(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    session = store.create_session("owner-a", "Private")
    store.append_messages("owner-a", session.id, "old", [{"role": "user", "content": "Original"}])
    original = store.load_messages("owner-a", session.id)
    with pytest.raises(SessionNotFoundError):
        store.replace_history("owner-b", session.id, [])
    assert store.load_messages("owner-a", session.id) == original


def test_replace_serializes_every_payload_before_deleting(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    session = store.create_session("owner-a", "Keep")
    store.append_messages("owner-a", session.id, "old", [{"role": "user", "content": "Original"}])
    original = store.load_messages("owner-a", session.id)
    with pytest.raises(TypeError):
        store.replace_history("owner-a", session.id, [
            {"role": "assistant", "content": "Summary", "_kind": "context_summary"},
            {"role": "user", "content": object()},
        ])
    assert store.load_messages("owner-a", session.id) == original


def test_replace_rolls_back_delete_and_partial_inserts_on_database_error(tmp_path):
    database = tmp_path / "sessions.db"
    store = SQLiteSessionStore(database)
    session = store.create_session("owner-a", "Keep")
    store.append_messages("owner-a", session.id, "old", [{"role": "user", "content": "Original"}])
    original = store.load_messages("owner-a", session.id)
    original_session = store.get_session("owner-a", session.id)
    # A real SQLite error after the first replacement row was inserted.
    with sqlite3.connect(database) as connection:
        connection.execute("""CREATE TRIGGER fail_second_insert BEFORE INSERT ON messages
            WHEN NEW.sequence = 2 BEGIN SELECT RAISE(ABORT, 'test insert failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="test insert failure"):
        store.replace_history("owner-a", session.id, [
            {"role": "assistant", "content": "Summary", "_kind": "context_summary"},
            {"role": "user", "content": "Recent"},
        ])
    assert store.load_messages("owner-a", session.id) == original
    assert store.get_session("owner-a", session.id) == original_session

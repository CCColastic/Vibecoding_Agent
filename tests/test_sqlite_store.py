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

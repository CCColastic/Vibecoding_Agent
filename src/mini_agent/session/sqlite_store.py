from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from mini_agent.session.models import Session, StoredMessage
from mini_agent.session.store import SessionNotFoundError


class SQLiteSessionStore:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_session(self, owner_id: str, title: str) -> Session:
        now = self._now()
        session = Session(
            id=str(uuid4()),
            owner_id=owner_id,
            title=title,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (id, owner_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.owner_id,
                    session.title,
                    self._serialize_time(session.created_at),
                    self._serialize_time(session.updated_at),
                ),
            )
        return session

    def get_session(self, owner_id: str, session_id: str) -> Session:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, owner_id, title, created_at, updated_at
                FROM sessions
                WHERE id = ? AND owner_id = ?
                """,
                (session_id, owner_id),
            ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return self._session_from_row(row)

    def list_sessions(self, owner_id: str) -> list[Session]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, owner_id, title, created_at, updated_at
                FROM sessions
                WHERE owner_id = ?
                ORDER BY updated_at DESC
                """,
                (owner_id,),
            ).fetchall()
        return [self._session_from_row(row) for row in rows]

    def load_messages(
        self, owner_id: str, session_id: str
    ) -> list[StoredMessage]:
        self.get_session(owner_id, session_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT session_id, turn_id, sequence, payload_json, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY sequence
                """,
                (session_id,),
            ).fetchall()
        return [
            StoredMessage(
                session_id=row["session_id"],
                turn_id=row["turn_id"],
                sequence=row["sequence"],
                payload=json.loads(row["payload_json"]),
                created_at=self._parse_time(row["created_at"]),
            )
            for row in rows
        ]

    def append_messages(
        self,
        owner_id: str,
        session_id: str,
        turn_id: str,
        messages: Sequence[dict[str, Any]],
    ) -> Session:
        if not messages:
            return self.get_session(owner_id, session_id)

        now = self._now()
        serialized_messages = [
            json.dumps(message, ensure_ascii=False) for message in messages
        ]
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, owner_id, title, created_at, updated_at
                FROM sessions
                WHERE id = ? AND owner_id = ?
                """,
                (session_id, owner_id),
            ).fetchone()
            if row is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            current_sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            connection.executemany(
                """
                INSERT INTO messages
                    (session_id, turn_id, sequence, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        turn_id,
                        current_sequence + index,
                        payload,
                        self._serialize_time(now),
                    )
                    for index, payload in enumerate(serialized_messages, start=1)
                ],
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (self._serialize_time(now), session_id),
            )

        return Session(
            id=row["id"],
            owner_id=row["owner_id"],
            title=row["title"],
            created_at=self._parse_time(row["created_at"]),
            updated_at=now,
        )

    def replace_history(
        self,
        owner_id: str,
        session_id: str,
        messages: Sequence[dict[str, Any]],
    ) -> Session:
        """Atomically replace this owner's Session history with a new snapshot."""
        now = self._now()
        rows_to_insert = []
        turn_id = str(uuid4())
        for sequence, message in enumerate(messages, start=1):
            if message.get("role") == "user" or message.get("_kind") == "context_summary":
                turn_id = str(uuid4())
            rows_to_insert.append((
                session_id, turn_id, sequence,
                json.dumps(message, ensure_ascii=False), self._serialize_time(now),
            ))

        # Prepare every payload before acquiring a write lock or deleting anything.
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT id, owner_id, title, created_at, updated_at
                   FROM sessions WHERE id = ? AND owner_id = ?""",
                (session_id, owner_id),
            ).fetchone()
            if row is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")
            connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            connection.executemany(
                """INSERT INTO messages
                   (session_id, turn_id, sequence, payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                rows_to_insert,
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ? AND owner_id = ?",
                (self._serialize_time(now), session_id, owner_id),
            )
        return Session(
            id=row["id"], owner_id=row["owner_id"], title=row["title"],
            created_at=self._parse_time(row["created_at"]), updated_at=now,
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_owner_updated
                ON sessions(owner_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id),
                    UNIQUE(session_id, sequence)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_sequence
                ON messages(session_id, sequence);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            owner_id=row["owner_id"],
            title=row["title"],
            created_at=SQLiteSessionStore._parse_time(row["created_at"]),
            updated_at=SQLiteSessionStore._parse_time(row["updated_at"]),
        )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _serialize_time(value: datetime) -> str:
        return value.isoformat()

    @staticmethod
    def _parse_time(value: str) -> datetime:
        return datetime.fromisoformat(value)

from __future__ import annotations

from typing import Any, Protocol, Sequence

from mini_agent.session.models import Session, StoredMessage


class SessionNotFoundError(LookupError):
    pass


class SessionStore(Protocol):
    def create_session(self, owner_id: str, title: str) -> Session: ...

    def get_session(self, owner_id: str, session_id: str) -> Session: ...

    def list_sessions(self, owner_id: str) -> list[Session]: ...

    def load_messages(
        self, owner_id: str, session_id: str
    ) -> list[StoredMessage]: ...

    def append_messages(
        self,
        owner_id: str,
        session_id: str,
        run_id: str,
        messages: Sequence[dict[str, Any]],
    ) -> Session: ...

    def replace_history(
        self,
        owner_id: str,
        session_id: str,
        messages: Sequence[dict[str, Any]],
    ) -> Session: ...

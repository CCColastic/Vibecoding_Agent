from __future__ import annotations

from mini_agent.runtime import AgentRuntime
from mini_agent.session.conversation import ActiveConversation
from mini_agent.session.models import Session
from mini_agent.session.store import SessionStore


class ConversationManager:
    __slots__ = ("_owner_id", "_runtime", "_store")

    def __init__(
        self,
        *,
        owner_id: str,
        runtime: AgentRuntime,
        store: SessionStore,
    ) -> None:
        self._owner_id = owner_id
        self._runtime = runtime
        self._store = store

    def new_conversation(self) -> ActiveConversation:
        return ActiveConversation(
            owner_id=self._owner_id,
            session_id=None,
            history=[],
            runtime=self._runtime,
            store=self._store,
        )

    def resume_conversation(self, session_id: str) -> ActiveConversation:
        session = self._store.get_session(self._owner_id, session_id)
        stored_messages = self._store.load_messages(self._owner_id, session.id)
        return ActiveConversation(
            owner_id=self._owner_id,
            session_id=session.id,
            history=[message.payload for message in stored_messages],
            runtime=self._runtime,
            store=self._store,
        )

    def list_sessions(self) -> list[Session]:
        return self._store.list_sessions(self._owner_id)

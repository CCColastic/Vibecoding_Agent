from __future__ import annotations

from dataclasses import dataclass

from mini_agent.runtime import AgentRuntime
from mini_agent.session.conversation import ActiveConversation
from mini_agent.session.models import Session
from mini_agent.session.store import SessionStore


@dataclass(frozen=True, slots=True)
class SessionService:
    runtime: AgentRuntime
    store: SessionStore

    def new_conversation(self, owner_id: str) -> ActiveConversation:
        return ActiveConversation(
            owner_id=owner_id,
            session_id=None,
            history=[],
            runtime=self.runtime,
            session_store=self.store,
        )

    def resume_conversation(
        self, owner_id: str, session_id: str
    ) -> ActiveConversation:
        session = self.store.get_session(owner_id, session_id)
        stored_messages = self.store.load_messages(owner_id, session.id)
        return ActiveConversation(
            owner_id=owner_id,
            session_id=session.id,
            history=[message.payload for message in stored_messages],
            runtime=self.runtime,
            session_store=self.store,
        )

    def list_sessions(self, owner_id: str) -> list[Session]:
        return self.store.list_sessions(owner_id)

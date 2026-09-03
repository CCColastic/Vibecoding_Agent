from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from mini_agent.models import RunResult
from mini_agent.runtime import AgentRuntime
from mini_agent.session.store import SessionStore


@dataclass(slots=True)
class ActiveConversation:
    owner_id: str
    session_id: str | None
    history: list[dict[str, Any]]
    runtime: AgentRuntime
    session_store: SessionStore

    async def send_message(self, user_input: str) -> RunResult:
        if not user_input.strip():
            raise ValueError("user_input must not be empty")

        if self.session_id is None:
            session = self.session_store.create_session(
                self.owner_id, self._title_from(user_input)
            )
            self.session_id = session.id

        result = await self.runtime.run(
            user_input=user_input,
            context_messages=self.history,
        )
        self.session_store.append_messages(
            owner_id=self.owner_id,
            session_id=self.session_id,
            turn_id=str(uuid4()),
            messages=result.new_messages,
        )
        self.history.extend(deepcopy(result.new_messages))
        return result

    @staticmethod
    def _title_from(user_input: str) -> str:
        return user_input.strip()[:30]

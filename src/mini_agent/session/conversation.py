from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from mini_agent.models import RunResult
from mini_agent.runtime import AgentRuntime
from mini_agent.session.store import SessionStore


class ActiveConversation:
    __slots__ = ("_history", "_owner_id", "_runtime", "_session_id", "_store")

    def __init__(
        self,
        *,
        owner_id: str,
        session_id: str | None,
        history: list[dict[str, Any]],
        runtime: AgentRuntime,
        store: SessionStore,
    ) -> None:
        self._owner_id = owner_id
        self._session_id = session_id
        self._history = history
        self._runtime = runtime
        self._store = store

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def send_message(self, user_input: str) -> RunResult:
        if not user_input.strip():
            raise ValueError("user_input must not be empty")

        if self._session_id is None:
            session = self._store.create_session(
                self._owner_id, self._title_from(user_input)
            )
            self._session_id = session.id

        result = await self._runtime.run(
            user_input=user_input,
            context_messages=self._history,
            run_id=str(uuid4()),
            session_id=self._session_id,
        )
        if result.status in {"compaction_error", "context_limit_exceeded"}:
            return result
        if result.compacted:
            self._store.replace_history(
                owner_id=self._owner_id,
                session_id=self._session_id,
                messages=result.messages,
            )
        else:
            self._store.append_messages(
                owner_id=self._owner_id,
                session_id=self._session_id,
                run_id=result.run_id,
                messages=result.new_messages,
            )
        self._history = deepcopy(result.messages)
        return result

    @staticmethod
    def _title_from(user_input: str) -> str:
        return user_input.strip()[:30]

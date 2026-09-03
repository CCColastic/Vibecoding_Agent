from typing import Any, Sequence

from mini_agent import AgentDefinition
from mini_agent.session import Session, SessionService, StoredMessage
from tests.fakes import FakeLLMClient


class CountingStore:
    def __init__(self, session: Session, messages: list[StoredMessage]) -> None:
        self.session = session
        self.messages = messages
        self.loads = 0
        self.appends = 0

    def get_session(self, owner_id: str, session_id: str) -> Session:
        return self.session

    def load_messages(self, owner_id: str, session_id: str) -> list[StoredMessage]:
        self.loads += 1
        return self.messages

    def list_sessions(self, owner_id: str) -> list[Session]:
        return [self.session]

    def append_messages(
        self,
        owner_id: str,
        session_id: str,
        turn_id: str,
        messages: Sequence[dict[str, Any]],
    ) -> Session:
        self.appends += 1
        return self.session


async def test_resume_loads_once_then_uses_active_history() -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    session = Session("session-1", "owner-a", "Greeting", now, now)
    stored = [
        StoredMessage(
            "session-1",
            "turn-1",
            1,
            {"role": "user", "content": "My name is Ada"},
            now,
        ),
        StoredMessage(
            "session-1",
            "turn-1",
            2,
            {"role": "assistant", "content": "Hello Ada"},
            now,
        ),
    ]
    store = CountingStore(session, stored)
    client = FakeLLMClient(
        [
            {"role": "assistant", "content": "Ada"},
            {"role": "assistant", "content": "Still Ada"},
        ]
    )
    runtime = AgentDefinition("Remember", []).create_runtime(
        llm_client=client
    )
    service = SessionService(runtime, store)

    conversation = service.resume_conversation("owner-a", "session-1")
    await conversation.send_message("What is my name?")
    await conversation.send_message("Are you sure?")

    assert store.loads == 1
    assert store.appends == 2
    assert conversation.runtime is runtime
    assert client.calls[0]["messages"][1:3] == [item.payload for item in stored]


def test_new_conversation_does_not_access_store() -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    session = Session("session-1", "owner-a", "Unused", now, now)
    store = CountingStore(session, [])
    runtime = AgentDefinition("Be helpful", []).create_runtime(
        llm_client=FakeLLMClient([])
    )
    service = SessionService(runtime, store)

    conversation = service.new_conversation("owner-a")

    assert conversation.session_id is None
    assert conversation.history == []
    assert store.loads == 0

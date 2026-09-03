from copy import deepcopy
from typing import Any, Sequence

import pytest

from mini_agent import AgentDefinition
from mini_agent.session import ActiveConversation, Session
from mini_agent.tools import CalculatorTool
from tests.fakes import FakeLLMClient, tool_call


class RecordingStore:
    def __init__(self, *, fail_append: bool = False) -> None:
        self.fail_append = fail_append
        self.created: list[tuple[str, str]] = []
        self.appended: list[list[dict[str, Any]]] = []
        self.session: Session | None = None

    def create_session(self, owner_id: str, title: str) -> Session:
        from datetime import datetime, timezone

        self.created.append((owner_id, title))
        now = datetime.now(timezone.utc)
        self.session = Session("session-1", owner_id, title, now, now)
        return self.session

    def append_messages(
        self,
        owner_id: str,
        session_id: str,
        turn_id: str,
        messages: Sequence[dict[str, Any]],
    ) -> Session:
        if self.fail_append:
            raise OSError("database unavailable")
        self.appended.append(deepcopy(list(messages)))
        assert self.session is not None
        return self.session


def make_runtime(client: FakeLLMClient):
    definition = AgentDefinition("Remember context", [CalculatorTool()])
    return definition.create_runtime(llm_client=client)


async def test_new_conversation_is_lazy_and_reuses_memory_history() -> None:
    client = FakeLLMClient(
        [
            {"role": "assistant", "content": "Hello Ada"},
            {"role": "assistant", "content": "Your name is Ada"},
        ]
    )
    store = RecordingStore()
    runtime = make_runtime(client)
    conversation = ActiveConversation("owner-a", None, [], runtime, store)

    assert store.created == []
    first = await conversation.send_message("My name is Ada")
    second = await conversation.send_message("What is my name?")

    assert conversation.session_id == "session-1"
    assert store.created[0] == ("owner-a", "My name is Ada")
    assert len(store.created) == 1
    assert second.steps_used == 1
    assert client.calls[1]["messages"] == [
        {"role": "system", "content": "Remember context"},
        *first.new_messages,
        {"role": "user", "content": "What is my name?"},
    ]
    assert conversation.history == first.new_messages + second.new_messages
    assert runtime.llm_client is client


async def test_tool_messages_are_available_to_follow_up() -> None:
    client = FakeLLMClient(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    tool_call(
                        "calculator",
                        '{"operation":"multiply","a":6,"b":7}',
                    )
                ],
            },
            {"role": "assistant", "content": "42"},
            {"role": "assistant", "content": "The previous result was 42"},
        ]
    )
    store = RecordingStore()
    conversation = ActiveConversation("owner-a", None, [], make_runtime(client), store)

    first = await conversation.send_message("Calculate 6 times 7")
    await conversation.send_message("What was the result?")

    assert first.tool_executions[0].result.content == 42
    second_call_messages = client.calls[2]["messages"]
    assert any(message.get("role") == "tool" for message in second_call_messages)


async def test_failed_persistence_does_not_update_memory() -> None:
    client = FakeLLMClient([{"role": "assistant", "content": "Answer"}])
    store = RecordingStore(fail_append=True)
    conversation = ActiveConversation("owner-a", None, [], make_runtime(client), store)

    with pytest.raises(OSError, match="database unavailable"):
        await conversation.send_message("Question")

    assert conversation.history == []


async def test_blank_message_is_rejected_without_creating_session() -> None:
    store = RecordingStore()
    conversation = ActiveConversation(
        "owner-a", None, [], make_runtime(FakeLLMClient([])), store
    )

    with pytest.raises(ValueError, match="must not be empty"):
        await conversation.send_message("   ")

    assert store.created == []

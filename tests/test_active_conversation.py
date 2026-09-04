from copy import deepcopy
from typing import Any, Sequence

import pytest

from mini_agent import AgentDefinition
from mini_agent.session import ActiveConversation, Session
from mini_agent.tools import CalculatorTool
from mini_agent.runtime_config import RuntimeConfig
from tests.fakes import FakeLLMClient, llm_response, message_payloads, tool_call


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
        run_id: str,
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
    conversation = ActiveConversation(
        owner_id="owner-a",
        session_id=None,
        history=[],
        runtime=make_runtime(client),
        store=store,
    )

    first = await conversation.send_message("Calculate 6 times 7")
    await conversation.send_message("What was the result?")

    assert first.tool_executions[0].result.content == 42
    second_call_messages = client.calls[2]["messages"]
    assert second_call_messages[1:-1] == message_payloads(first.new_messages)
    assert store.created[0][1] == "Calculate 6 times 7"
    assert len(store.created) == 1


async def test_failed_persistence_does_not_update_memory() -> None:
    client = FakeLLMClient(
        [
            {"role": "assistant", "content": "Answer"},
            {"role": "assistant", "content": "Retry answer"},
        ]
    )
    store = RecordingStore(fail_append=True)
    conversation = ActiveConversation(
        owner_id="owner-a",
        session_id=None,
        history=[],
        runtime=make_runtime(client),
        store=store,
    )

    with pytest.raises(OSError, match="database unavailable"):
        await conversation.send_message("Question")

    store.fail_append = False
    await conversation.send_message("Retry")

    assert client.calls[1]["messages"] == [
        {"role": "system", "content": "Remember context"},
        {"role": "user", "content": "Retry"},
    ]


async def test_blank_message_is_rejected_without_creating_session() -> None:
    store = RecordingStore()
    conversation = ActiveConversation(
        owner_id="owner-a",
        session_id=None,
        history=[],
        runtime=make_runtime(FakeLLMClient([])),
        store=store,
    )

    with pytest.raises(ValueError, match="must not be empty"):
        await conversation.send_message("   ")

    assert store.created == []


async def test_budget_stop_saves_only_completed_run_messages() -> None:
    call = tool_call(
        "calculator", '{"operation":"add","a":1,"b":2}'
    )
    client = FakeLLMClient([
        llm_response(
            {"content": None, "tool_calls": [call]},
            prompt_tokens=90,
            completion_tokens=10,
        )
    ])
    runtime = AgentDefinition("Use tools", [CalculatorTool()]).create_runtime(
        llm_client=client,
        runtime_config=RuntimeConfig(max_chat_usage=100),
    )
    store = RecordingStore()
    conversation = ActiveConversation(
        owner_id="owner-a", session_id=None, history=[],
        runtime=runtime, store=store,
    )
    result = await conversation.send_message("Calculate")
    assert result.status == "token_budget_exceeded"
    assert message_payloads(store.appended[0]) == [
        {"role": "user", "content": "Calculate"}
    ]

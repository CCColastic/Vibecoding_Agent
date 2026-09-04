from __future__ import annotations

from copy import deepcopy
from typing import Any

from mini_agent.llm import LLMResponse, LLMUsage


def llm_response(
    message: dict[str, Any],
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    finish_reason: str | None = "stop",
    usage_available: bool = True,
) -> LLMResponse:
    usage = None if not usage_available else LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return LLMResponse(message=message, usage=usage, finish_reason=finish_reason)

def message_payloads(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expected persisted/model content without the local Run identity."""
    return [{key: value for key, value in message.items() if key != "_run_id"}
            for message in messages]


class FakeLLMClient:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    async def llm_call(
        self,
        *,
        purpose: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
                "purpose": purpose,
            }
        )
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, LLMResponse):
            return response
        return llm_response(response)


class FailingLLMClient:
    async def llm_call(
        self,
        *,
        purpose: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        raise RuntimeError("secret internal detail")


def tool_call(
    name: str,
    arguments: str,
    *,
    call_id: str = "call-1",
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }

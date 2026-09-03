from __future__ import annotations

from copy import deepcopy
from typing import Any


class FakeLLMClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "model": model,
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
            }
        )
        return next(self._responses)


class FailingLLMClient:
    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
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

from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]: ...

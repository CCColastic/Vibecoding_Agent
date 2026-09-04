from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol


LLMPurpose = Literal["chat", "compaction"]


@dataclass(frozen=True, slots=True)
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class LLMResponse:
    message: dict[str, Any]
    usage: LLMUsage | None
    finish_reason: str | None


class LLMClient(Protocol):
    async def llm_call(
        self,
        *,
        purpose: LLMPurpose,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse: ...

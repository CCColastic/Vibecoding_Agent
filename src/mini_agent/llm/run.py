from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from mini_agent.llm.base import LLMClient, LLMResponse, LLMPurpose
from mini_agent.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class TokenBudgetExceeded(RuntimeError):
    def __init__(self, purpose: LLMPurpose, used: int, limit: int) -> None:
        self.purpose = purpose
        self.used = used
        self.limit = limit
        super().__init__(f"{purpose} token budget exceeded: used={used}, limit={limit}")


@dataclass(slots=True)
class RunUsage:
    chat_tokens: int = 0
    compaction_tokens: int = 0
    complete: bool = True
    last_usage: Any = None
    last_finish_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.chat_tokens + self.compaction_tokens

    def used(self, purpose: LLMPurpose) -> int:
        return self.chat_tokens if purpose == "chat" else self.compaction_tokens

    def add(self, purpose: LLMPurpose, tokens: int) -> None:
        if purpose == "chat":
            self.chat_tokens += tokens
        else:
            self.compaction_tokens += tokens


class RunLLM:
    """Run-scoped budget and usage accounting around a shared LLM client."""

    def __init__(
        self, *, client: LLMClient, config: RuntimeConfig, usage: RunUsage
    ) -> None:
        self.client = client
        self.config = config
        self.usage = usage

    async def call(
        self,
        *,
        purpose: LLMPurpose,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        self.ensure_available(purpose)

        try:
            response = await self.client.llm_call(
                purpose=purpose, messages=messages, tools=tools
            )
        except BaseException:
            self.usage.complete = False
            raise
        self.usage.last_usage = response.usage
        self.usage.last_finish_reason = response.finish_reason
        if response.usage is None:
            self.usage.complete = False
            logger.warning("LLM usage unavailable purpose=%s", purpose)
        else:
            self.usage.add(purpose, response.usage.total_tokens)
        return response

    def limit(self, purpose: LLMPurpose) -> int:
        return (
            self.config.max_chat_usage
            if purpose == "chat"
            else self.config.max_compaction_usage
        )

    def exhausted(self, purpose: LLMPurpose) -> bool:
        return self.usage.used(purpose) >= self.limit(purpose)

    def ensure_available(self, purpose: LLMPurpose) -> None:
        if self.exhausted(purpose):
            raise TokenBudgetExceeded(
                purpose, self.usage.used(purpose), self.limit(purpose)
            )

    def budget_error(self, purpose: LLMPurpose) -> str:
        return str(TokenBudgetExceeded(
            purpose, self.usage.used(purpose), self.limit(purpose)
        ))

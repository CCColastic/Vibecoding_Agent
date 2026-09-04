from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from mini_agent.llm.base import LLMResponse, LLMPurpose, LLMUsage
from mini_agent.llm.config import ModelConfig


class DeepSeekClient:
    def __init__(self, config: ModelConfig | None = None) -> None:
        load_dotenv()
        required = ("API_KEY", "BASE_URL")
        if config is None:
            required += ("MODEL",)
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise ValueError(
                "Missing required environment variables: " + ", ".join(missing)
            )

        self.config = config or ModelConfig(model=os.environ["MODEL"])
        self._client = AsyncOpenAI(
            api_key=os.environ["API_KEY"],
            base_url=os.environ["BASE_URL"],
            max_retries=0,
        )

    async def llm_call(
        self,
        *,
        purpose: LLMPurpose,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        max_tokens = (
            self.config.chat_max_output_tokens
            if purpose == "chat"
            else self.config.compaction_max_output_tokens
        )
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if tools:
            request["tools"] = tools

        for attempt in range(self.config.max_attempts):
            try:
                response = await self._client.chat.completions.create(**request)
                break
            except (APIConnectionError, APIStatusError) as exc:
                retryable = (
                    isinstance(exc, APIConnectionError)
                    or exc.status_code == 429
                    or 500 <= exc.status_code < 600
                )
                if not retryable or attempt == self.config.max_attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

        choice = response.choices[0]
        usage = response.usage
        normalized_usage = None if usage is None else LLMUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
        return LLMResponse(
            message=choice.message.model_dump(exclude_none=True),
            usage=normalized_usage,
            finish_reason=choice.finish_reason,
        )

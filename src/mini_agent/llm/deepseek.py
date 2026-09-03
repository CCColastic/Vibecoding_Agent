from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI


class DeepSeekClient:
    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not resolved_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url="https://api.deepseek.com",
        )

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if tools:
            request["tools"] = tools

        response = await self._client.chat.completions.create(**request)
        return response.choices[0].message.model_dump(exclude_none=True)

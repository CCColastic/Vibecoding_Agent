from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AsyncOpenAI


class DeepSeekClient:
    def __init__(self) -> None:
        load_dotenv()
        required = ("API_KEY", "BASE_URL", "MODEL")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise ValueError(
                "Missing required environment variables: " + ", ".join(missing)
            )

        self._model = os.environ["MODEL"]
        self._client = AsyncOpenAI(
            api_key=os.environ["API_KEY"],
            base_url=os.environ["BASE_URL"],
            max_retries=0,
        )

    async def llm_call(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if tools:
            request["tools"] = tools

        for attempt in range(3):
            try:
                response = await self._client.chat.completions.create(**request)
                break
            except (APIConnectionError, APIStatusError) as exc:
                retryable = (
                    isinstance(exc, APIConnectionError)
                    or exc.status_code == 429
                    or 500 <= exc.status_code < 600
                )
                if not retryable or attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)

        return response.choices[0].message.model_dump(exclude_none=True)

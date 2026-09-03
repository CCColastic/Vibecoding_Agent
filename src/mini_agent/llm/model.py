from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI


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

        response = await self._client.chat.completions.create(**request)
        return response.choices[0].message.model_dump(exclude_none=True)

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from mini_agent.models import ToolResult


class Tool(Protocol):
    name: str
    description: str
    arguments_model: type[BaseModel]

    async def execute(self, **arguments: Any) -> ToolResult: ...

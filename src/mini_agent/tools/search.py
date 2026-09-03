from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mini_agent.models import ToolResult


class SearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, description="The search query")


class SearchTool:
    name = "search"
    description = "Search for information using a mock implementation."
    arguments_model = SearchArguments

    async def execute(self, *, query: str) -> ToolResult:
        return ToolResult(ok=True, content="mock result")

from __future__ import annotations

from typing import Any, Iterable

from pydantic import BaseModel, ValidationError

from mini_agent.tools.base import Tool


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    @staticmethod
    def validate_tool(tool: Tool) -> None:
        if not isinstance(tool.name, str) or not tool.name:
            raise ValueError("Tool name must be a non-empty string")
        if not isinstance(tool.description, str) or not tool.description:
            raise ValueError(f"Tool '{tool.name}' must have a description")
        arguments_model = getattr(tool, "arguments_model", None)
        if not isinstance(arguments_model, type) or not issubclass(
            arguments_model, BaseModel
        ):
            raise ValueError(
                f"Tool '{tool.name}' must define a Pydantic arguments_model"
            )

    def register(self, tool: Tool) -> None:
        self.validate_tool(tool)
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.arguments_model.model_json_schema(),
                },
            }
            for tool in self._tools.values()
        ]

    def validate_arguments(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        try:
            validated = tool.arguments_model.model_validate(arguments)
        except ValidationError as exc:
            raise ValueError(f"Invalid arguments for tool '{name}': {exc}") from exc
        return validated.model_dump()

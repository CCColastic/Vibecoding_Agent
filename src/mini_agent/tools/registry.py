from __future__ import annotations

import json
from typing import Any, Iterable

from pydantic import BaseModel, ValidationError

from mini_agent.models import ToolExecution, ToolResult
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

    async def execute_tool_call(self, tool_call: Any) -> ToolExecution:
        call_id = "unknown"
        name = "unknown"
        arguments: dict[str, Any] = {}
        try:
            if not isinstance(tool_call, dict):
                raise ValueError("Tool call must be an object")
            call_id = str(tool_call.get("id") or "unknown")
            function = tool_call.get("function")
            if not isinstance(function, dict):
                raise ValueError("Tool call is missing function data")
            name = str(function.get("name") or "unknown")
            raw_arguments = function.get("arguments", "{}")
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")

            arguments = self.validate_arguments(name, arguments)
            tool = self.get(name)
            if tool is None:
                raise ValueError(f"Unknown tool: {name}")
            result = await tool.execute(**arguments)
            if not isinstance(result, ToolResult):
                raise TypeError("Tool must return ToolResult")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            result = ToolResult(ok=False, content=str(exc))
        except Exception as exc:
            result = ToolResult(
                ok=False,
                content=f"Tool '{name}' failed: {type(exc).__name__}",
            )

        return ToolExecution(
            tool_call_id=call_id,
            name=name,
            arguments=arguments,
            result=result,
        )

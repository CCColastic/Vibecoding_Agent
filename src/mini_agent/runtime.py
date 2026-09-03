from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mini_agent.llm.base import LLMClient
from mini_agent.models import RunResult, ToolExecution, ToolResult
from mini_agent.tools.registry import ToolRegistry


@dataclass(slots=True)
class AgentRuntime:
    system_prompt: str
    model: str
    registry: ToolRegistry
    llm_client: LLMClient
    max_steps: int = 8

    async def run(self, user_input: str) -> RunResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        executions: list[ToolExecution] = []

        for step in range(1, self.max_steps + 1):
            try:
                assistant_message = await self.llm_client.chat_completion(
                    model=self.model,
                    messages=messages,
                    tools=self.registry.schemas(),
                )
            except Exception as exc:
                return RunResult(
                    status="llm_error",
                    final_answer=None,
                    messages=messages,
                    steps_used=step,
                    tool_executions=executions,
                    error=f"LLM request failed: {type(exc).__name__}",
                )

            content = assistant_message.get("content")
            tool_calls = assistant_message.get("tool_calls") or []
            stored_assistant = {"role": "assistant", "content": content}
            if tool_calls:
                stored_assistant["tool_calls"] = tool_calls
            messages.append(stored_assistant)

            if tool_calls:
                for tool_call in tool_calls:
                    execution = await self._execute_tool_call(tool_call)
                    executions.append(execution)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": execution.tool_call_id,
                            "content": self._tool_content(execution.result),
                        }
                    )
                continue

            if isinstance(content, str) and content.strip():
                return RunResult(
                    status="completed",
                    final_answer=content,
                    messages=messages,
                    steps_used=step,
                    tool_executions=executions,
                )

            return RunResult(
                status="llm_protocol_error",
                final_answer=None,
                messages=messages,
                steps_used=step,
                tool_executions=executions,
                error="LLM response contained neither text nor tool calls",
            )

        return RunResult(
            status="max_steps_exceeded",
            final_answer=None,
            messages=messages,
            steps_used=self.max_steps,
            tool_executions=executions,
            error=f"Agent reached max_steps={self.max_steps}",
        )

    async def _execute_tool_call(self, tool_call: Any) -> ToolExecution:
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

            arguments = self.registry.validate_arguments(name, arguments)
            tool = self.registry.get(name)
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

    @staticmethod
    def _tool_content(result: ToolResult) -> str:
        payload = {"ok": result.ok, "content": result.content}
        try:
            return json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError):
            return json.dumps(
                {"ok": False, "content": "Tool returned non-serializable content"}
            )

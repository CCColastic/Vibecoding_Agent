from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

from mini_agent.llm.base import LLMClient
from mini_agent.models import RunResult, RunState, RunStatus, ToolResult
from mini_agent.tools.registry import ToolRegistry


@dataclass(slots=True)
class AgentRuntime:
    system_prompt: str
    registry: ToolRegistry
    llm_client: LLMClient
    max_steps: int = 8

    async def run(
        self,
        user_input: str,
        context_messages: Sequence[dict[str, Any]] = (),
    ) -> RunResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *deepcopy(list(context_messages)),
        ]
        state = RunState(messages=messages, new_messages_start=len(messages))
        state.messages.append({"role": "user", "content": user_input})

        for step in range(1, self.max_steps + 1):
            state.step = step
            try:
                assistant_message = await self.llm_client.llm_call(
                    messages=state.messages,
                    tools=self.registry.schemas(),
                )
            except Exception as exc:
                return self._result(
                    state,
                    status="llm_error",
                    final_answer=None,
                    error=f"LLM request failed: {type(exc).__name__}",
                )

            content = assistant_message.get("content")
            tool_calls = assistant_message.get("tool_calls") or []
            stored_assistant = {"role": "assistant", "content": content}
            if tool_calls:
                stored_assistant["tool_calls"] = tool_calls
            state.messages.append(stored_assistant)

            if tool_calls:
                for tool_call in tool_calls:
                    execution = await self.registry.execute_tool_call(tool_call)
                    state.tool_executions.append(execution)
                    state.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": execution.tool_call_id,
                            "content": self._tool_content(execution.result),
                        }
                    )
                continue

            if isinstance(content, str) and content.strip():
                return self._result(
                    state,
                    status="completed",
                    final_answer=content,
                )

            return self._result(
                state,
                status="llm_protocol_error",
                final_answer=None,
                error="LLM response contained neither text nor tool calls",
            )

        return self._result(
            state,
            status="max_steps_exceeded",
            final_answer=None,
            error=f"Agent reached max_steps={self.max_steps}",
        )

    @staticmethod
    def _result(
        state: RunState,
        *,
        status: RunStatus,
        final_answer: str | None,
        error: str | None = None,
    ) -> RunResult:
        return RunResult(
            status=status,
            final_answer=final_answer,
            messages=state.messages,
            new_messages=state.messages[state.new_messages_start :],
            steps_used=state.step,
            tool_executions=state.tool_executions,
            error=error,
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

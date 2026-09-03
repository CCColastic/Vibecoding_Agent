from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

from mini_agent.context import (
    CompactionError,
    ContextCompactor,
    ContextLimitExceeded,
    ContextPolicy,
)
from mini_agent.context.compactor import request_messages
from mini_agent.llm.base import LLMClient
from mini_agent.models import RunResult, RunState, RunStatus, ToolResult
from mini_agent.tools.registry import ToolRegistry


@dataclass(slots=True)
class AgentRuntime:
    system_prompt: str
    registry: ToolRegistry
    llm_client: LLMClient
    max_steps: int = 8
    compactor: ContextCompactor | None = None

    def __post_init__(self) -> None:
        if self.compactor is None:
            self.compactor = ContextCompactor(
                llm_client=self.llm_client, policy=ContextPolicy()
            )

    async def run(
        self,
        user_input: str,
        context_messages: Sequence[dict[str, Any]] = (),
    ) -> RunResult:
        state = RunState(messages=deepcopy(list(context_messages)))
        self._append_message(state, {"role": "user", "content": user_input})
        tools = self.registry.schemas()

        for step in range(1, self.max_steps + 1):
            try:
                assert self.compactor is not None
                prepared = await self.compactor.prepare(
                    messages=state.messages,
                    current_messages=state.new_messages,
                    system_prompt=self.system_prompt,
                    tools=tools,
                )
            except CompactionError as exc:
                return self._result(
                    state, status="compaction_error", final_answer=None, error=str(exc)
                )
            except ContextLimitExceeded as exc:
                return self._result(
                    state, status="context_limit_exceeded", final_answer=None, error=str(exc)
                )
            state.messages = prepared.messages
            state.compacted = state.compacted or prepared.compacted
            state.step = step
            try:
                assistant_message = await self.llm_client.llm_call(
                    messages=request_messages(self.system_prompt, state.messages),
                    tools=tools,
                )
            except Exception as exc:
                return self._result(
                    state,
                    status="llm_error",
                    final_answer=None,
                    error=f"LLM request failed: {type(exc).__name__}",
                )

            if not isinstance(assistant_message, dict):
                return self._result(
                    state, status="llm_protocol_error", final_answer=None,
                    error="LLM response must be a message object",
                )
            content = assistant_message.get("content")
            tool_calls = assistant_message.get("tool_calls") or []
            stored_assistant = {"role": "assistant", "content": content}
            if tool_calls:
                stored_assistant["tool_calls"] = tool_calls
                self._append_message(state, stored_assistant)
                for tool_call in tool_calls:
                    execution = await self.registry.execute_tool_call(tool_call)
                    state.tool_executions.append(execution)
                    self._append_message(
                        state,
                        {
                            "role": "tool",
                            "tool_call_id": execution.tool_call_id,
                            "content": self._tool_content(execution.result),
                        }
                    )
                continue

            if isinstance(content, str) and content.strip():
                self._append_message(state, stored_assistant)
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
    def _append_message(state: RunState, message: dict[str, Any]) -> None:
        state.messages.append(message)
        state.new_messages.append(message)

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
            new_messages=state.new_messages,
            steps_used=state.step,
            tool_executions=state.tool_executions,
            error=error,
            compacted=state.compacted,
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

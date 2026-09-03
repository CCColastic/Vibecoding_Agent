from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Sequence

from mini_agent.context import (
    CompactionError,
    ContextCompactor,
    ContextLimitExceeded,
    ContextPolicy,
)
from mini_agent.context.compactor import request_messages
from mini_agent.llm.base import LLMClient
from mini_agent.models import RunResult, RunState, RunStatus, ToolExecution, ToolResult
from mini_agent.tools.registry import ToolRegistry
from mini_agent.trace import TraceEvent, TraceEventType, TraceRecorder, resolve_run_id

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentRuntime:
    system_prompt: str
    registry: ToolRegistry
    llm_client: LLMClient
    max_steps: int = 8
    compactor: ContextCompactor | None = None
    trace_recorder: TraceRecorder | None = None

    def __post_init__(self) -> None:
        if self.compactor is None:
            self.compactor = ContextCompactor(
                llm_client=self.llm_client, policy=ContextPolicy()
            )

    async def run(
        self,
        user_input: str,
        context_messages: Sequence[dict[str, Any]] = (),
        *,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> RunResult:
        state = RunState(
            messages=deepcopy(list(context_messages)),
            run_id=resolve_run_id(run_id), session_id=session_id,
        )
        started = perf_counter()
        status = "unhandled_error"
        error = None
        try:
            self._append_message(state, {"role": "user", "content": user_input})
            self._emit(state, "user.input", {"content": user_input})
            result = await self._run_loop(state)
            status, error = result.status, result.error
            return result
        except asyncio.CancelledError:
            status, error = "cancelled", "Run cancelled"
            raise
        except BaseException as exc:
            error = f"Run interrupted: {type(exc).__name__}"
            raise
        finally:
            self._emit(state, "run.end", {
                "status": status, "steps_used": state.step, "error": error,
                "duration_ms": (perf_counter() - started) * 1000,
            })

    async def _run_loop(self, state: RunState) -> RunResult:
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
            if prepared.compacted:
                state.messages[0] = {**state.messages[0], "_run_id": state.run_id}
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
            self._emit(state, "assistant.output", {"content": content, "tool_calls": tool_calls})
            stored_assistant = {"role": "assistant", "content": content}
            if tool_calls:
                stored_assistant["tool_calls"] = tool_calls
                self._append_message(state, stored_assistant)
                for tool_call in tool_calls:
                    execution = await self._trace_tool_call(state, tool_call)
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
        message = {**message, "_run_id": state.run_id}
        state.messages.append(message)
        state.new_messages.append(message)

    def _emit(self, state: RunState, event: TraceEventType, data: dict[str, Any]) -> None:
        if self.trace_recorder is None:
            return
        state.trace_sequence += 1
        try:
            self.trace_recorder.emit(TraceEvent(
                timestamp=datetime.now(timezone.utc), run_id=state.run_id,
                session_id=state.session_id, sequence=state.trace_sequence,
                step=state.step, event=event, data=data,
            ))
        except Exception as exc:
            logger.warning("Trace recorder failed: %s", type(exc).__name__)

    async def _trace_tool_call(self, state: RunState, tool_call: Any) -> ToolExecution:
        call = tool_call if isinstance(tool_call, dict) else {}
        function = call.get("function")
        function = function if isinstance(function, dict) else {}
        identity = {
            "tool_call_id": str(call.get("id") or "unknown"),
            "name": str(function.get("name") or "unknown"),
        }
        self._emit(state, "tool.start", {
            **identity, "raw_arguments": function.get("arguments", "{}"),
        })
        started = perf_counter()
        end: dict[str, Any] = {**identity, "arguments": None}
        try:
            execution = await self.registry.execute_tool_call(tool_call)
            end.update(
                tool_call_id=execution.tool_call_id, name=execution.name,
                arguments=execution.arguments,
                result={"ok": execution.result.ok, "content": execution.result.content},
            )
            return execution
        except asyncio.CancelledError:
            end["result"] = {"ok": False, "content": "Tool call cancelled"}
            raise
        except BaseException as exc:
            end["result"] = {"ok": False, "content": f"Tool call interrupted: {type(exc).__name__}"}
            raise
        finally:
            end["duration_ms"] = (perf_counter() - started) * 1000
            self._emit(state, "tool.end", end)

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
            run_id=state.run_id,
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

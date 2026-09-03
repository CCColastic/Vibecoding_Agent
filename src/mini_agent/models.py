from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    content: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolExecution:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    result: ToolResult


@dataclass(slots=True)
class RunState:
    messages: list[dict[str, Any]]
    new_messages_start: int
    step: int = 0
    tool_executions: list[ToolExecution] = field(default_factory=list)


RunStatus = Literal[
    "completed",
    "max_steps_exceeded",
    "llm_error",
    "llm_protocol_error",
]


@dataclass(frozen=True, slots=True)
class RunResult:
    status: RunStatus
    final_answer: str | None
    messages: list[dict[str, Any]]
    new_messages: list[dict[str, Any]]
    steps_used: int
    tool_executions: list[ToolExecution]
    error: str | None = None

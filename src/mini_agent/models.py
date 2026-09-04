from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from mini_agent.llm.run import RunUsage


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
    run_id: str
    session_id: str | None = None
    trace_sequence: int = 0
    new_messages: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    tool_executions: list[ToolExecution] = field(default_factory=list)
    compacted: bool = False
    usage: RunUsage = field(default_factory=RunUsage)


RunStatus = Literal[
    "completed",
    "max_steps_exceeded",
    "llm_error",
    "llm_protocol_error",
    "compaction_error",
    "context_limit_exceeded",
    "token_budget_exceeded",
    "output_truncated",
]


@dataclass(frozen=True, slots=True)
class RunResult:
    status: RunStatus
    final_answer: str | None
    messages: list[dict[str, Any]]
    new_messages: list[dict[str, Any]]
    steps_used: int
    tool_executions: list[ToolExecution]
    run_id: str
    error: str | None = None
    compacted: bool = False
    chat_token_usage: int = 0
    compaction_token_usage: int = 0
    token_usage: int = 0
    usage_complete: bool = True

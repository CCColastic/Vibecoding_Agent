from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    context_limit: int = 1_000_000
    trigger_ratio: float = 0.7
    keep_recent_turns: int = 4
    max_summary_chars: int = 8_000
    output_reserve: int = 8_192

    def __post_init__(self) -> None:
        if self.context_limit <= 0 or not 0 <= self.output_reserve < self.context_limit:
            raise ValueError("context_limit must exceed output_reserve >= 0")
        if not 0 < self.trigger_ratio < 1:
            raise ValueError("trigger_ratio must be between 0 and 1")
        if self.keep_recent_turns < 1 or self.max_summary_chars < 1:
            raise ValueError("keep_recent_turns and max_summary_chars must be positive")


@dataclass(frozen=True, slots=True)
class PreparedContext:
    messages: list[dict[str, Any]]
    compacted: bool = False


class CompactionError(RuntimeError):
    """A summary could not be generated or validated."""


class ContextLimitExceeded(RuntimeError):
    """The protected context cannot fit the configured estimate budget."""

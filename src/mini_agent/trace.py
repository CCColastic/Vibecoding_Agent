from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

TraceEventType = Literal[
    "user.input", "assistant.output", "tool.start", "tool.end", "run.end",
]


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return str(uuid4())
    try:
        parsed = UUID(run_id)
    except (ValueError, TypeError, AttributeError):
        raise ValueError("run_id must be a UUID4") from None
    if parsed.version != 4:
        raise ValueError("run_id must be a UUID4")
    return str(parsed)


@dataclass(frozen=True, slots=True)
class TraceEvent:
    timestamp: datetime
    run_id: str
    session_id: str | None
    sequence: int
    step: int
    event: TraceEventType
    data: dict[str, Any]


class TraceRecorder:
    """Best-effort, indented JSON recording; no current-Run state is stored here."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def emit(self, event: TraceEvent) -> None:
        temporary_path: Path | None = None
        try:
            run_id = resolve_run_id(event.run_id)
            payload = {
                "timestamp": event.timestamp.isoformat(),
                "run_id": run_id,
                "session_id": event.session_id,
                "sequence": event.sequence,
                "step": event.step,
                "event": event.event,
                "data": event.data,
            }
            destination = self.directory / f"{run_id}.json"
            events = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else []
            if not isinstance(events, list):
                raise ValueError("Trace file must contain a JSON array")
            events.append(payload)
            content = json.dumps(
                events, ensure_ascii=False, indent=2,
                default=lambda value: {"unserializable_type": type(value).__name__},
            )
            self.directory.mkdir(parents=True, exist_ok=True)
            # Replace a complete JSON document so interrupted writes keep the old trace.
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory,
                prefix=f".{run_id}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(content + "\n")
            temporary_path.replace(destination)
        except Exception as exc:
            # Do not expose paths, payloads, or provider details from an exception.
            logger.warning("Trace write failed: %s", type(exc).__name__)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("Trace temporary file cleanup failed: %s", type(exc).__name__)

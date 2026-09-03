from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class OwnerProfile:
    owner_id: str


@dataclass(frozen=True, slots=True)
class Session:
    id: str
    owner_id: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class StoredMessage:
    session_id: str
    run_id: str
    sequence: int
    payload: dict[str, Any]
    created_at: datetime

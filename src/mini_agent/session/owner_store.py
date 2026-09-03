from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

from mini_agent.session.models import OwnerProfile


class OwnerConfigError(ValueError):
    pass


class LocalOwnerStore:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._config_path = data_dir / "config.json"

    def get_or_create(self) -> OwnerProfile:
        if self._config_path.exists():
            return self._read()

        profile = OwnerProfile(owner_id=str(uuid4()))
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(
            json.dumps({"owner_id": profile.owner_id}, indent=2),
            encoding="utf-8",
        )
        return profile

    def _read(self) -> OwnerProfile:
        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            owner_id = data["owner_id"]
            UUID(owner_id)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise OwnerConfigError(
                f"Invalid owner configuration: {self._config_path}"
            ) from exc
        return OwnerProfile(owner_id=owner_id)

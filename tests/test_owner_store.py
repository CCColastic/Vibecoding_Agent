import json

import pytest

from mini_agent.session import LocalOwnerStore, OwnerConfigError


def test_owner_id_is_created_once_and_reused(tmp_path) -> None:
    store = LocalOwnerStore(tmp_path)

    first = store.get_or_create()
    second = LocalOwnerStore(tmp_path).get_or_create()

    assert first == second
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8")) == {
        "owner_id": first.owner_id
    }


def test_invalid_owner_config_fails_without_replacing_it(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text("not json", encoding="utf-8")

    with pytest.raises(OwnerConfigError, match="Invalid owner configuration"):
        LocalOwnerStore(tmp_path).get_or_create()

    assert config.read_text(encoding="utf-8") == "not json"

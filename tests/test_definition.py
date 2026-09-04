from dataclasses import FrozenInstanceError

import pytest

from mini_agent import AgentDefinition
from mini_agent.tools import CalculatorTool, SearchTool
from tests.fakes import FakeLLMClient
from mini_agent.runtime_config import RuntimeConfig


def test_definition_is_immutable_and_freezes_tool_list() -> None:
    source_tools = [CalculatorTool()]
    definition = AgentDefinition("Be helpful", source_tools)
    source_tools.append(SearchTool())

    assert len(definition.tools) == 1
    assert isinstance(definition.tools, tuple)
    with pytest.raises(FrozenInstanceError):
        definition.system_prompt = "another prompt"  # type: ignore[misc]


def test_duplicate_tool_names_fail_during_definition() -> None:
    with pytest.raises(ValueError, match="Duplicate tool name: calculator"):
        AgentDefinition(
            "Be helpful",
            [CalculatorTool(), CalculatorTool()],
        )

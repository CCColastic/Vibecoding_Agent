import pytest

from mini_agent.tools import CalculatorTool, SearchTool, ToolRegistry
from tests.fakes import tool_call


def test_registry_returns_deepseek_function_schemas() -> None:
    registry = ToolRegistry([CalculatorTool(), SearchTool()])

    schemas = registry.schemas()

    assert [item["function"]["name"] for item in schemas] == [
        "calculator",
        "search",
    ]
    assert schemas[0]["function"]["parameters"]["additionalProperties"] is False
    assert registry.validate_arguments(
        "calculator", {"operation": "add", "a": 1, "b": 2}
    ) == {"operation": "add", "a": 1, "b": 2}


@pytest.mark.parametrize("arguments", [
    {"operation": "add", "a": 1},
    {"operation": "add", "a": "1", "b": 2},
    {"operation": "add", "a": 1, "b": 2, "extra": True},
], ids=["missing", "wrong-type", "extra"])
def test_registry_rejects_invalid_arguments(arguments):
    registry = ToolRegistry([CalculatorTool()])
    with pytest.raises(ValueError, match="Invalid arguments"):
        registry.validate_arguments("calculator", arguments)


@pytest.mark.parametrize("name,arguments,error", [
    ("calculator", "not-json", "Expecting value"),
    ("missing", "{}", "Unknown tool"),
])
async def test_registry_turns_invalid_calls_into_tool_errors(name, arguments, error):
    registry = ToolRegistry([CalculatorTool()])

    execution = await registry.execute_tool_call(
        tool_call(name, arguments)
    )

    assert execution.result.ok is False
    assert error in execution.result.content

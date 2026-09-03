import pytest

from mini_agent.tools import CalculatorTool, SearchTool, ToolRegistry


def test_registry_returns_deepseek_function_schemas() -> None:
    registry = ToolRegistry([CalculatorTool(), SearchTool()])

    schemas = registry.schemas()

    assert [item["function"]["name"] for item in schemas] == [
        "calculator",
        "search",
    ]
    assert schemas[0]["function"]["parameters"]["additionalProperties"] is False


def test_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry([CalculatorTool()])

    with pytest.raises(ValueError, match="Duplicate tool name"):
        registry.register(CalculatorTool())


def test_registry_validates_arguments() -> None:
    registry = ToolRegistry([CalculatorTool()])

    validated = registry.validate_arguments(
        "calculator", {"operation": "add", "a": 1, "b": 2}
    )
    assert validated == {"operation": "add", "a": 1, "b": 2}
    with pytest.raises(ValueError, match="Invalid arguments"):
        registry.validate_arguments("calculator", {"operation": "add", "a": 1})


def test_registry_uses_strict_pydantic_validation() -> None:
    registry = ToolRegistry([CalculatorTool()])

    with pytest.raises(ValueError, match="Invalid arguments"):
        registry.validate_arguments(
            "calculator", {"operation": "add", "a": "1", "b": 2}
        )


def test_registry_rejects_extra_arguments() -> None:
    registry = ToolRegistry([SearchTool()])

    with pytest.raises(ValueError, match="Invalid arguments"):
        registry.validate_arguments("search", {"query": "x", "extra": True})


def test_registry_rejects_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Unknown tool"):
        registry.validate_arguments("missing", {})

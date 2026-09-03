import pytest

from mini_agent.tools import CalculatorTool, SearchTool, WeatherTool


@pytest.mark.parametrize(
    ("operation", "a", "b", "expected"),
    [
        ("add", 8, 2, 10),
        ("subtract", 8, 2, 6),
        ("multiply", 8, 2, 16),
        ("divide", 8, 2, 4),
    ],
)
async def test_calculator_operations(
    operation: str,
    a: int,
    b: int,
    expected: int | float,
) -> None:
    result = await CalculatorTool().execute(operation=operation, a=a, b=b)

    assert result.ok is True
    assert result.content == expected


async def test_calculator_rejects_division_by_zero() -> None:
    result = await CalculatorTool().execute(operation="divide", a=8, b=0)

    assert result.ok is False
    assert result.content == "Division by zero is not allowed"


async def test_search_returns_fixed_mock_result() -> None:
    result = await SearchTool().execute(query="anything")

    assert result.ok is True
    assert result.content == "mock result"


async def test_weather_returns_mock_data_for_location() -> None:
    result = await WeatherTool().execute(location="Hong Kong")

    assert result.ok is True
    assert result.content == {
        "location": "Hong Kong",
        "condition": "sunny",
        "temperature_c": 25,
        "source": "mock",
    }

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mini_agent.models import ToolResult


class WeatherArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    location: str = Field(min_length=1, description="The location to check")


class WeatherTool:
    name = "weather"
    description = "Get mock weather data for a location."
    arguments_model = WeatherArguments

    async def execute(self, *, location: str) -> ToolResult:
        return ToolResult(
            ok=True,
            content={
                "location": location,
                "condition": "sunny",
                "temperature_c": 25,
                "source": "mock",
            },
        )

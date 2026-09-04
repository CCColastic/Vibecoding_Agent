from __future__ import annotations

import asyncio

from mini_agent import AgentDefinition
from mini_agent.tools import CalculatorTool, SearchTool, WeatherTool
from mini_agent.runtime_config import RuntimeConfig


async def main() -> None:
    definition = AgentDefinition(
        system_prompt=(
            "You are a helpful assistant. Use the available tools when useful, "
            "then give the user a concise final answer."
        ),
        tools=[CalculatorTool(), SearchTool(), WeatherTool()],
    )
    runtime = definition.create_runtime(runtime_config=RuntimeConfig(max_steps=8))
    result = await runtime.run("Calculate 12 multiplied by 7.")
    print(result.final_answer or f"{result.status}: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())

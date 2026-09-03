from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mini_agent.models import ToolResult


class CalculatorArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="The arithmetic operation to perform"
    )
    a: int | float = Field(description="The first number")
    b: int | float = Field(description="The second number")


class CalculatorTool:
    name = "calculator"
    description = "Perform one basic arithmetic operation on two numbers."
    arguments_model = CalculatorArguments

    async def execute(
        self,
        *,
        operation: str,
        a: int | float,
        b: int | float,
    ) -> ToolResult:
        if operation == "add":
            value = a + b
        elif operation == "subtract":
            value = a - b
        elif operation == "multiply":
            value = a * b
        elif operation == "divide":
            if b == 0:
                return ToolResult(ok=False, content="Division by zero is not allowed")
            value = a / b
        else:
            return ToolResult(ok=False, content=f"Unsupported operation: {operation}")
        return ToolResult(ok=True, content=value)

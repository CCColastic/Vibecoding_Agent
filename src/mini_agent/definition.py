from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mini_agent.context import ContextCompactor, ContextPolicy
from mini_agent.llm import DeepSeekClient, LLMClient
from mini_agent.runtime import AgentRuntime
from mini_agent.tools import Tool, ToolRegistry


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    system_prompt: str
    tools: Sequence[Tool]

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        frozen_tools = tuple(self.tools)
        seen_names: set[str] = set()
        for tool in frozen_tools:
            ToolRegistry.validate_tool(tool)
            if tool.name in seen_names:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            seen_names.add(tool.name)
        object.__setattr__(self, "tools", frozen_tools)

    def create_runtime(
        self,
        *,
        max_steps: int = 8,
        llm_client: LLMClient | None = None,
        context_policy: ContextPolicy | None = None,
    ) -> AgentRuntime:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        registry = ToolRegistry(self.tools)
        client = llm_client if llm_client is not None else DeepSeekClient()
        return AgentRuntime(
            system_prompt=self.system_prompt,
            registry=registry,
            llm_client=client,
            max_steps=max_steps,
            compactor=ContextCompactor(
                llm_client=client,
                policy=context_policy if context_policy is not None else ContextPolicy(),
            ),
        )

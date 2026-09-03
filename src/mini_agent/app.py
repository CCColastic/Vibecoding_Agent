from __future__ import annotations

import os
from pathlib import Path

from mini_agent import AgentDefinition
from mini_agent.llm import DeepSeekClient, LLMClient
from mini_agent.session import (
    ConversationManager,
    LocalOwnerStore,
    SQLiteSessionStore,
)
from mini_agent.tools import CalculatorTool, SearchTool, WeatherTool


def _default_data_dir() -> Path:
    configured = os.getenv("MINI_AGENT_DATA_DIR")
    return Path(configured) if configured else Path.home() / ".mini_agent"


def _build_default_definition() -> AgentDefinition:
    return AgentDefinition(
        system_prompt=(
            "You are a helpful assistant. Use the available tools when useful, "
            "then give the user a concise final answer."
        ),
        tools=[CalculatorTool(), SearchTool(), WeatherTool()],
    )


def build_conversation_manager(
    *,
    data_dir: Path | None = None,
    llm_client: LLMClient | None = None,
) -> ConversationManager:
    resolved_data_dir = data_dir or _default_data_dir()
    owner = LocalOwnerStore(resolved_data_dir).get_or_create()
    session_store = SQLiteSessionStore(resolved_data_dir / "sessions.db")
    definition = _build_default_definition()
    client = llm_client if llm_client is not None else DeepSeekClient()
    runtime = definition.create_runtime(llm_client=client)
    return ConversationManager(
        owner_id=owner.owner_id,
        runtime=runtime,
        store=session_store,
    )

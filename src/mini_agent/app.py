from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from mini_agent import AgentDefinition, AgentRuntime
from mini_agent.llm import DeepSeekClient, LLMClient
from mini_agent.session import (
    LocalOwnerStore,
    OwnerProfile,
    SQLiteSessionStore,
    SessionService,
)
from mini_agent.tools import CalculatorTool, SearchTool, WeatherTool


@dataclass(frozen=True, slots=True)
class Application:
    owner: OwnerProfile
    definition: AgentDefinition
    llm_client: LLMClient
    runtime: AgentRuntime
    session_store: SQLiteSessionStore
    session_service: SessionService


def default_data_dir() -> Path:
    configured = os.getenv("MINI_AGENT_DATA_DIR")
    return Path(configured) if configured else Path.home() / ".mini_agent"


def build_default_definition() -> AgentDefinition:
    return AgentDefinition(
        system_prompt=(
            "You are a helpful assistant. Use the available tools when useful, "
            "then give the user a concise final answer."
        ),
        tools=[CalculatorTool(), SearchTool(), WeatherTool()],
    )


def build_application(
    *,
    data_dir: Path | None = None,
    llm_client: LLMClient | None = None,
) -> Application:
    resolved_data_dir = data_dir or default_data_dir()
    owner = LocalOwnerStore(resolved_data_dir).get_or_create()
    session_store = SQLiteSessionStore(resolved_data_dir / "sessions.db")
    definition = build_default_definition()
    client = llm_client if llm_client is not None else DeepSeekClient()
    runtime = definition.create_runtime(llm_client=client)
    session_service = SessionService(runtime=runtime, store=session_store)
    return Application(
        owner=owner,
        definition=definition,
        llm_client=client,
        runtime=runtime,
        session_store=session_store,
        session_service=session_service,
    )

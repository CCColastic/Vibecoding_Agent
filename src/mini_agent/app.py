from __future__ import annotations

from pathlib import Path

from mini_agent import AgentDefinition
from mini_agent.llm import DeepSeekClient, LLMClient
from mini_agent.session import (
    ConversationManager,
    LocalOwnerStore,
    SQLiteSessionStore,
)
from mini_agent.tools import CalculatorTool, SearchTool, WeatherTool
from mini_agent.trace import TraceRecorder


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2]


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
    trace_enabled: bool = True,
) -> ConversationManager:
    resolved_data_dir = data_dir or _default_data_dir()
    owner = LocalOwnerStore(resolved_data_dir).get_or_create()
    session_store = SQLiteSessionStore(resolved_data_dir / "sessions.db")
    definition = _build_default_definition()
    client = llm_client if llm_client is not None else DeepSeekClient()
    recorder = TraceRecorder(resolved_data_dir / "traces") if trace_enabled else None
    runtime = definition.create_runtime(llm_client=client, trace_recorder=recorder)
    return ConversationManager(
        owner_id=owner.owner_id,
        runtime=runtime,
        store=session_store,
    )

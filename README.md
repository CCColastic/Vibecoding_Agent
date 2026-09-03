# mini_agent

`mini_agent` is a minimal asynchronous ReAct-style agent runtime backed by the
DeepSeek Chat Completions API. It supports native tool calls, persistent
sessions, consecutive follow-up questions, a hard `max_steps` limit, and three
small example tools.

## Requirements

- Python 3.12+
- A DeepSeek API key for the real example

## Install

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

Set the API key in your shell:

```powershell
$env:DEEPSEEK_API_KEY="your-api-key"
```

The library reads the environment variable directly; it does not load `.env`
files automatically.

## Run the CLI

Start a new conversation:

```powershell
.venv\Scripts\mini-agent new
```

The Session is created only after the first non-empty message. Continue typing
to ask follow-up questions, or enter `/exit` to leave the Session.

List and resume an existing Session:

```powershell
.venv\Scripts\mini-agent sessions
```

The CLI creates one local Owner ID in `~/.mini_agent/config.json` and stores
Sessions in `~/.mini_agent/sessions.db`. Set `MINI_AGENT_DATA_DIR` to use a
different directory.

## Run the single-turn example

```powershell
.venv\Scripts\python examples\run_agent.py
```

```python
import asyncio

from mini_agent import AgentDefinition
from mini_agent.tools import CalculatorTool, SearchTool, WeatherTool


async def main() -> None:
    definition = AgentDefinition(
        system_prompt="Use tools when useful, then answer clearly.",
        model="deepseek-chat",
        tools=[CalculatorTool(), SearchTool(), WeatherTool()],
    )
    result = await definition.create_runtime(max_steps=8).run(
        "What is 12 multiplied by 7?"
    )
    print(result.final_answer or result.error)


asyncio.run(main())
```

## Design

`AgentDefinition` is immutable configuration. The application creates one
DeepSeek client, one tool registry, and one `AgentRuntime`, then reuses them for
the whole CLI process. The Runtime never owns Session state.

For every `run(user_input, context_messages)`, the Runtime creates a temporary
`RunState`; `steps_used` therefore restarts for every user turn. A response
containing tool calls is validated, executed in order, appended to the message
list, and sent back to the model. The resulting `new_messages` contain only the
current user message and the assistant/tool messages produced for that turn.

`ActiveConversation` owns the in-memory history for one Session. A new
conversation starts with an empty history and creates its Session on the first
message. A resumed conversation loads SQLite history once. After every turn,
only `new_messages` are appended to SQLite and then to the in-memory history,
so later questions can refer to prior answers and tool results without reading
the full history from SQLite again.

The current version intentionally has no context compaction or Session-level
concurrent writing. It loads the complete history when a Session is resumed.

## Test

```powershell
.venv\Scripts\python -m pytest
```

Tests inject a deterministic fake LLM client, so the default suite does not use
the network or consume API credit.

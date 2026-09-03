# mini_agent

`mini_agent` is a minimal asynchronous ReAct-style agent runtime backed by the
DeepSeek Chat Completions API. It supports native tool calls, persistent
sessions, consecutive follow-up questions, rolling context summaries, a hard
`max_steps` limit, and three small example tools.

## Requirements

- Python 3.12+
- A DeepSeek API key for the real example

## Install

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

Set the model configuration in your shell or `.env` file:

```powershell
$env:API_KEY="your-api-key"
$env:BASE_URL="https://api.deepseek.com"
$env:MODEL="deepseek-chat"
```

The model adapter calls `load_dotenv()` and reads `API_KEY`, `BASE_URL`, and
`MODEL`. Shell environment values take precedence over values in `.env`.

Each `DeepSeekClient.llm_call()` makes at most three attempts (the initial request
plus two retries), with asynchronous waits of 1 and 2 seconds. Only connection
errors, SDK timeouts, HTTP 429, and HTTP 5xx are retried. Other HTTP errors,
invalid successful responses, and cancellation are not retried. SDK built-in
retries are disabled (`max_retries=0`) to avoid multiplying attempts. After the
last failure, the original exception is passed to the existing Runtime or
compaction error handler. Both normal and summary requests share this behavior;
retries do not consume extra Agent steps, append messages, or re-execute tools.

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

The composition root returns an Owner-scoped `ConversationManager`. Callers can
create, list, and resume conversations without receiving the Owner ID, Runtime,
LLM client, or SQLite store:

```python
from mini_agent.app import build_conversation_manager

conversations = build_conversation_manager()
conversation = conversations.new_conversation()
result = await conversation.send_message("Hello")
```

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
        tools=[CalculatorTool(), SearchTool(), WeatherTool()],
    )
    result = await definition.create_runtime(max_steps=8).run(
        "What is 12 multiplied by 7?"
    )
    print(result.final_answer or result.error)


asyncio.run(main())
```

## Design

`AgentDefinition` is immutable configuration containing only the system prompt
and tools. Model name and DeepSeek connection settings belong to the model
adapter and come from the environment. The application creates one DeepSeek
client, one tool registry, and one `AgentRuntime`, then reuses them for the
whole CLI process. The Runtime never owns Session state.

For every `run(user_input, context_messages)`, the Runtime creates a temporary
`RunState`; `steps_used` therefore restarts for every user turn. A response
containing tool calls is validated, executed in order, appended to the message
list, and sent back to the model. The resulting `new_messages` contain only the
current user message and the assistant/tool messages produced for that turn.

`ActiveConversation` owns the in-memory history for one Session. A new
conversation starts with an empty history and creates its Session on the first
message. A resumed conversation loads SQLite effective history once. Without
compaction, `new_messages` are appended to SQLite; after compaction, the entire
effective history is replaced in one transaction. The in-memory history is
updated only after a successful commit, so later questions can refer to prior
answers and tool results without reading SQLite again. Its public interface contains only
`session_id` and `send_message()`; persistence and Runtime details stay private.

`RunState.messages` and `RunResult.messages` contain effective history without
the system prompt. `new_messages` explicitly accumulates only this Run's user,
assistant, and tool messages; it never contains a generated summary.

## Context compaction and memory recall

Before every normal LLM call (including calls after tool execution), the Runtime
checks `len(json.dumps({"messages": ..., "tools": ...}, ensure_ascii=False)) / 4`.
The estimate includes the system prompt and tool schemas, but excludes local
summary metadata. The defaults assume a 1,000,000-token window and trigger
compaction at 70%; this is application configuration, not automatic model-limit
discovery. Character counting is only a heuristic, especially for Chinese, and
cannot guarantee that the provider will accept the request.

The `ContextCompactor` reuses the same LLM client and stores no Session state.
It summarizes the previous summary plus older whole turns, keeping exactly
`keep_recent_turns` recent turns intact (four by default). If fewer turns exist,
all are protected and there are no older turns to summarize. Retention is never
reduced to fit the budget. The current Run is separately protected, and tool
calls/results remain paired. Summary requests use
`tools=[]`, are not Agent steps, and never recursively call the Runtime.

The effective history has this shape:

```python
[
    {"role": "assistant", "content": "历史摘要：...", "_kind": "context_summary"},
    # Recent complete user / assistant / tool messages, followed by this Run.
]
```

There is at most one summary. It is historical data, not a system instruction.
`_kind` is persisted locally and removed before sending messages to the model.
The system prompt is prepended only for LLM requests. On Session resume, this
summary and the retained messages are loaded once; on subsequent questions they
are recalled from memory automatically. No retrieval index or archive lookup is
used. A later compaction merges the existing summary rather than stacking new
summaries on top of it.

Configure the policy when creating a Runtime:

```python
from mini_agent import ContextPolicy

runtime = definition.create_runtime(
    context_policy=ContextPolicy(
        context_limit=1_000_000,
        trigger_ratio=0.7,
        keep_recent_turns=4,
        max_summary_chars=8_000,
        output_reserve=8_192,
    ),
)
```

`max_summary_chars` limits only generated summary text, not retained turns or the
current Run; no maximum-length placeholder is reserved. A short historical-summary
label is added locally. `output_reserve` reduces the estimated input budget;
it does not change the provider's output-token setting. Small test limits must
also use a smaller reserve and summary length. Summary responses are accepted
only if they are non-empty text without tool calls, fit the summary length,
reduce the estimated request size, and leave it below the trigger and input
budget. Summary requests themselves must fit the estimated input budget.

An invalid/failed summary returns `compaction_error`; an uncompressible context
returns `context_limit_exceeded`. Both stop the Run without saving any of its
history or updating the active memory, even if an earlier compaction in the Run
succeeded. A newly created empty Session may remain. Invalid summary content is
not regenerated automatically; transient request failures use the bounded client
retries described above. No chunking, silent truncation, or over-budget fallback
is performed. Ordinary
LLM failures and `max_steps` still save partial Run history as before. Previously
executed tool side effects cannot be undone by refusing to save messages.

### SQLite snapshot replacement

Compaction does **not** preserve an archive. `replace_history(owner_id,
session_id, messages)` receives the entire new effective history, including this
Run. It serializes all payloads first, starts a transaction, verifies ownership,
deletes all message rows for that Session, inserts the new snapshot, updates the
Session timestamp, and commits. Any database failure rolls back the deletion
and inserts. It never deletes the Session itself or affects another Session.
The caller must not append `new_messages` after replacement.

The existing tables are reused without migration. Messages are renumbered from
1 and receive new turn-group IDs and snapshot timestamps; tool-call IDs inside
payloads are preserved. Session identity, title, and creation time are unchanged.
These message records are a current context snapshot, not an original audit
trail. Summarized source text is no longer available through the application:
exact quotations, recovery of omitted facts, and original message timestamps
cannot be guaranteed. Concurrent writers to the same Session remain unsupported.

Compaction logs use Python logging (`mini_agent.context.compactor`) and contain
event names, estimated sizes, retained-turn counts, and safe error types, not
full messages. Enable INFO logging in an embedding application to see successful
compactions. The CLI retains its existing interaction and error display.

See [AI prompts and problem-solving notes](docs/AI_PROMPTS.md) for the summary
prompt contract and implementation decisions.

## Test

```powershell
.venv\Scripts\python -m pytest
```

Tests inject a deterministic fake LLM client, so the default suite does not use
the network or consume API credit.

# mini_agent

从零实现的最小可用 ReAct 风格 Agent，使用真实 DeepSeek Chat Completions 接口，
支持工具调用、连续追问、独立 Session、滚动摘要和执行 trace。

核心 Agent Runtime 自行实现，不依赖 LangGraph、OpenHands、OpenClaw 等 Agent 框架。
`openai` 仅作为兼容接口的请求 SDK，工具参数 Schema 和校验使用 Pydantic，
持久化使用标准库 SQLite。

## 1. 运行方式

### 安装

需要 Python 3.12+，以及可用的 DeepSeek API Key。以下命令在项目根目录执行，
以 Windows PowerShell 为例：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

不需要激活虚拟环境，后续直接使用其中的可执行文件。

### 配置模型

首次运行时，从模板创建 `.env`；已有文件不会被覆盖：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

编辑项目根目录的 `.env`：

```dotenv
API_KEY=your-api-key
BASE_URL=https://api.deepseek.com
MODEL=your-model-name
```

必须将 `API_KEY` 和 `MODEL` 替换成实际值；模型需支持 Chat Completions 工具调用。
参数名与 [.env.example](.env.example) 一致，不使用 `DEEPSEEK_API_KEY` 等其他名称。
模型名称和连接设置由 `DeepSeekClient` 读取，不在 `AgentDefinition` 中配置。

客户端通过 `load_dotenv()` 加载配置，已有系统环境变量优先于 `.env`。
`.env` 已被 Git 忽略，不要提交真实密钥。

### 新建对话与连续追问

```powershell
.venv\Scripts\mini-agent.exe new
```

进入后可连续输入，例如：

```text
You: 我叫小明，请记住。
You: 我叫什么？
You: 用计算器计算 12 乘以 7。
You: 再把刚才的结果加上 10。
You: 查询上海天气，并说明数据是否真实。
You: /exit
```

首次非空消息到来时才创建 Session。空白输入被忽略；`/exit` 或输入时按
`Ctrl+C` 可离开，不删除已保存的历史。中断尚未完成的 Run 不保证保存其消息。

CLI 显示 Session ID，并为每个返回结果显示 `Run: <UUID>`，用于定位执行记录。

### 恢复历史 Session

```powershell
.venv\Scripts\mini-agent.exe sessions
```

程序列出当前 Owner 的 Session，按最近更新时间排序。输入编号并回车，随后继续对话。
两个窗口分别使用 `new`，会获得同一 Owner 下的两个独立 Session；恢复时选择对应编号即可。
目前不支持两个窗口同时写入同一个 Session。

### 关闭 trace / 运行单轮示例

```powershell
.venv\Scripts\mini-agent.exe new --no-trace
.venv\Scripts\mini-agent.exe sessions --no-trace
.venv\Scripts\python.exe examples\run_agent.py
```

单轮示例直接创建 Runtime，不创建持久化 Session，默认也不记录 trace。

### 本地数据位置

按上述 editable 安装方式运行时，默认数据目录是项目根目录，由应用源码位置确定，
而不是启动命令时的工作目录：

- `config.json`：首次生成并复用的本地 `owner_id`。
- `sessions.db`：Session 信息与有效历史。
- `traces/<run_id>.json`：默认启用的逐 Run 执行记录。

这些文件均被 Git 忽略。用户不需要手动提供 Owner ID 或 Session ID。
Owner 只是本地身份，不是登录或用户认证系统。

当前 CLI 不读取 `MINI_AGENT_DATA_DIR`；旧版用户目录下的数据也不会自动迁移。
代码调用时可通过 `build_conversation_manager(data_dir=Path(...))` 指定其他位置。

## 2. 系统设计

### 模块职责

- `app.py`：应用组装入口，创建 Owner、Store、Definition、Client、Runtime 和 trace recorder。
- `AgentDefinition`：只保存 `system_prompt` 和工具列表；创建 Runtime 时注册工具、注入依赖。
- `AgentRuntime`：执行 LLM/Tool Loop，管理单次 Run，不保存跨问题的 Session 状态。
- `DeepSeekClient`：发送真实模型请求，统一处理有限次数的请求重试。
- `ToolRegistry`：注册工具、生成 Schema、解析参数、校验并执行工具。
- `ConversationManager`：限定在一个 Owner 下，负责新建、列出和恢复 Session。
- `ActiveConversation`：持有当前 Session 的内存历史，协调运行与保存。
- `SQLiteSessionStore`：校验 Session 所有权，加载、追加或替换有效历史。
- `ContextCompactor`：估算上下文大小，生成滚动摘要，不持有 Session 状态。
- `TraceRecorder`：按 Run 写入执行事件，不参与 memory 召回。

应用内复用同一个 Client、Runtime、ToolRegistry、Compactor 和 Store。
每次提问只创建新的 `RunState`，`step` 从 0 开始，默认最多进行 8 次正常 Agent LLM 调用。
Session 状态只属于各自的 ActiveConversation，因此复用 Runtime 不会混合不同会话。

### Agent Loop

1. 接收用户输入，复制 Session 有效历史，创建 RunState 和新的 UUID4 `run_id`。
2. 追加当前 user 消息，在调用 LLM 前检查是否需要压缩。
3. 将 system prompt、有效历史、当前 Run 消息与 tools Schema 发给模型。
4. 若返回 `tool_calls`，按顺序解析 JSON 参数、通过 Pydantic 校验、执行工具。
5. 将 assistant 的工具调用及对应 tool 结果追加到上下文，再进入下一步 LLM 调用。
6. 若没有工具调用且 `content` 非空，作为最终答案返回；达到步数上限或发生错误则停止。

本项目使用原生工具调用字段决策，不要求模型生成可解析的 `Thought/Action` 文本。
当前请求关闭 thinking；内部思考不保存、不召回，也不写入 trace。
同一次响应中即使既有文本又有工具调用，也先执行工具，再继续 Loop。

### 三个工具

- `calculator`：必填 `operation`、`a`、`b`，支持
  `add / subtract / multiply / divide`；除零返回错误，不使用 `eval`。
- `search`：必填非空 `query`，固定返回 `"mock result"`。
- `weather`：必填非空 `location`，返回该地区的模拟天气：
  `sunny`、`25℃`、`source="mock"`，不是实时天气。

每个工具提供名称、描述、Pydantic 参数模型和异步 `execute()`。
Registry 将参数模型转换成 JSON Schema 交给 LLM；工具选择由模型决定，
不是根据用户输入关键词硬编码路由。参数采用严格校验，拒绝多余字段。
未知工具、非法参数和工具执行错误会转成工具结果，让 Agent 有机会继续回答。

### Run 数据

`RunState.messages` 保存本次运行使用的有效历史；`new_messages` 单独累积本轮新增的
user、assistant 和 tool 消息，不依赖历史切片起点，也不包含摘要。

`RunResult` 返回 `status`、`final_answer`、`messages`、`new_messages`、
`steps_used`、`tool_executions`、`run_id`、`compacted` 和 `error`。
其中 `messages` 不含 system prompt，可在持久化成功后成为下一轮的有效历史。

`owner_id` 表示本地归属，`session_id` 表示可持续的独立对话，`run_id` 表示一次执行。
同一 Session 的每次提问使用新的 Run ID；内部请求重试仍属于同一个 Run。

## 3. Memory：召回时机与放置方式

这里的 memory 是“有效历史 + 可选的滚动摘要”，不是向量数据库或自动长期记忆。
LLM 不会因为 Client 长期复用就自动记住之前的请求；连续追问依靠每次显式传入相关历史。

### 什么时候召回？

1. **新建对话**：ActiveConversation 从空 history 开始，不读取历史、不立即创建 Session。
2. **恢复 Session**：验证 Owner 后，从 SQLite 加载一次有效历史，创建 ActiveConversation。
3. **连续追问**：直接使用 ActiveConversation 的内存 history，不再从 SQLite 重新加载。
4. **同一 Run 内继续调用工具**：使用 RunState 中刚追加的工具调用和结果，不查询数据库。
5. **每轮结束并保存成功后**：更新内存 history，供下一次提问使用。

因此，纯对话追问可以看到前一轮 user/assistant；工具追问还可以看到完整的工具调用、
参数、tool 结果和最终回答。更早的内容被压缩后，只能通过摘要中的信息继续引用。

### 放在请求的什么位置？

发给 LLM 的 messages 顺序如下；这是结构示意，实际近期历史可以包含多轮：

```python
[
    {"role": "system", "content": system_prompt},
    {"role": "assistant", "content": "历史摘要：用户目标、约束、关键结果……"},
    # 最近未压缩的完整轮次：
    # user -> assistant(tool_calls) -> tool -> assistant
    {"role": "user", "content": current_user_input},
    # 当前 Run 已产生的 assistant / tool 消息继续追加在这里
]
```

未发生压缩时省略摘要。tools Schema 通过请求的独立 `tools` 参数传入，不放进 messages。

- system prompt 每次请求时注入，不存入 Session 历史。
- 摘要使用 assistant 消息放在历史开头，是历史资料，不是更高优先级的 system 指令。
- 当前 Run 的消息保持原样，不纳入本轮压缩。
- 内部消息带有 `_run_id`；摘要还带有 `_kind="context_summary"`。
  正常请求和摘要请求发送前都会移除这两个内部标记。
- SQLite 将 Run ID 放在 `messages.run_id` 列；恢复时再补回 `_run_id`。
  `_kind` 保存在摘要的 JSON payload 中。

trace 文件不加入上述 messages，也没有从 trace 自动检索或恢复遗漏事实的逻辑。

## 4. Context 压缩与保存

### 触发与摘要

每次正常 LLM 调用之前（包括工具执行之后），计算：

```python
estimated_tokens = len(
    json.dumps({"messages": request_messages, "tools": tools}, ensure_ascii=False)
) / 4
```

默认使用 `context_limit=1_000_000`、`trigger_ratio=0.7`，
即估算达到 700,000 tokens 时尝试压缩。1M 是本项目的配置假设，
不是自动检测到的模型能力；应按所选模型调整。字符数 / 4 对中文等内容可能低估，
不能保证实际请求不超限。

压缩过程：

1. 从历史中取出已有摘要，按 user 消息划分完整轮次。
2. 固定保留最近 `keep_recent_turns` 轮（默认 4 轮）及当前 Run，不能拆开工具调用与结果。
3. 将“已有摘要 + 更早轮次”交给同一个 LLM Client，使用独立摘要 Prompt 和 `tools=[]`。
4. 保留目标、事实与约束、已完成事项、关键工具结果及未完成事项；保留 mock 标记。
5. 摘要检查通过后，用一条新摘要取代旧摘要与更早历史。

最近轮次数量不会为了满足预算而自动减少。历史不足或等于保留轮数时全部受保护；
如果仍超出预算，就返回错误，而不是截断用户输入。

摘要必须是非空文本、不包含工具调用、不超过长度限制；替换后的完整请求必须确实变短，
并低于触发线和输入预算。摘要请求本身也要满足输入预算，不做分块或递归摘要。
摘要内容不合格不会自动重新生成；底层临时请求故障仍适用客户端重试。

### 配置示例

```python
from mini_agent import AgentDefinition, ContextPolicy
from mini_agent.tools import CalculatorTool, SearchTool, WeatherTool

definition = AgentDefinition(
    system_prompt="按需使用工具，并清楚说明结果。",
    tools=[CalculatorTool(), SearchTool(), WeatherTool()],
)
runtime = definition.create_runtime(
    max_steps=8,
    context_policy=ContextPolicy(
        context_limit=1_000_000,
        trigger_ratio=0.7,
        keep_recent_turns=4,
        max_summary_chars=8_000,
        output_reserve=8_192,
    ),
)
```

`max_summary_chars` 只限制生成的摘要文本，不限制保留轮次的长度，
也不按最大摘要长度预占空间。`output_reserve` 用于估算输入预算
`context_limit - output_reserve`，不改变模型接口的输出 token 设置。
摘要调用及请求重试不消耗正常 Agent 的 `max_steps`。

### 如何保存到 SQLite？

每条 message 对应 `messages` 表的一条记录，按 Session 内的 `sequence` 排序。

- **未压缩**：只追加 `RunResult.new_messages`。
- **已压缩**：调用 `replace_history()`，整体替换该 Session 的有效历史，
  包含新摘要、保留轮次和当前 Run 消息；替换后不能再次追加本轮消息。

替换时先序列化所有新消息，再在同一个事务中校验 Owner、删除目标 Session 的全部旧消息、
插入新列表并更新 Session 时间。无需计算旧消息有多少条；中途失败全部回滚，
不删除 Session 本身，不影响其他 Session。

数据库提交成功后才更新内存 history。替换后消息序号从 1 重排、消息时间为快照写入时间，
但保留轮次原来的 Run ID 和 tool-call ID 不变；新摘要属于生成它的 Run。

旧数据库中的 `turn_id` 列会在初始化时迁移为 `run_id`，保留原有值；
不会凭空补出过去未记录的 trace。

**数据保留限制：** SQLite 只保留有效上下文，被摘要覆盖的原文不再出现在 Session 历史中，
不能保证逐字引用或找回遗漏细节。trace 是独立记录，可能仍含这些原文，
但不会被 Agent 自动召回，也不会随着压缩自动删除。

## 5. 异常处理与执行 trace

### 异常与重试

Run 结果状态包括：

- `completed`：获得最终答案。
- `max_steps_exceeded`：达到正常 Agent LLM 调用次数上限。
- `llm_error`：模型请求最终失败。
- `llm_protocol_error`：响应没有有效文本或工具调用。
- `compaction_error`：摘要请求最终失败，或摘要内容不合格。
- `context_limit_exceeded`：无法在保留规则下满足上下文预算。

后两种状态不会写回本轮历史，也不会更新会话内存，即使该 Run 更早时曾成功压缩。
普通 LLM 错误或步数耗尽仍保存已产生的部分消息。数据库保存失败同样不更新内存。
已经执行的工具副作用不会因为停止或保存失败而撤销。

`DeepSeekClient.llm_call()` 最多尝试 3 次：首次请求加 2 次重试，异步等待 1 秒、2 秒。
只重试连接错误、SDK 超时、HTTP 429 和 5xx；认证失败、非法参数、取消操作和无效响应不重试。
SDK 自带重试已关闭，耗尽后重新抛出最后一次异常，由 Runtime 或 Compactor 处理。
重试不重复追加消息或执行工具。

### Trace

CLI 默认将事件保存为 `traces/<run_id>.json`，每个文件是可读的 UTF-8 JSON 数组：

- `user.input`：当前用户输入。
- `assistant.output`：正常模型响应的文本与工具调用，不含内部思考。
- `tool.start` / `tool.end`：工具名称、调用 ID、参数、结果或错误、耗时。
- `run.end`：结束状态、步数、错误和总耗时。

事件包含 UTC 时间、Run ID、Session ID、事件序号和当前 step。每次写入通过临时文件替换，
尽量保留上一次完整记录；写入失败只产生警告，不改变 Agent 执行结果。
正常结束、可处理错误和取消操作都会尝试记录结束事件，但强制终止进程仍可能留下不完整 trace。

`run.end` 表示 Runtime 结束，不代表 SQLite 提交成功。trace 不随 Session 事务回滚。

Trace 会记录用户原文、assistant 输出、工具参数和结果，可能包含敏感信息。
文件已被 Git 忽略，没有自动清理；`--no-trace` 仅关闭后续记录，不删除已有文件。
压缩日志另由 `mini_agent.context.compactor` 的 Python logger 输出，不包含完整历史。

## 6. 在代码中使用连续对话

```python
import asyncio

from mini_agent.app import build_conversation_manager


async def main():
    conversations = build_conversation_manager(trace_enabled=False)
    conversation = conversations.new_conversation()

    first = await conversation.send_message("我叫小明。")
    second = await conversation.send_message("我叫什么？")
    print(first.final_answer or first.error)
    print(second.final_answer or second.error)

    resumed = conversations.resume_conversation(conversation.session_id)
    third = await resumed.send_message("继续刚才的对话。")
    print(third.final_answer or third.error)


if __name__ == "__main__":
    asyncio.run(main())
```

需要持久化和自动追问历史时，使用 ActiveConversation。
直接调用 `runtime.run()` 不会自动创建 Session、保存历史或读取上一次调用；
调用者需要自己传入 `context_messages`。独立 Runtime 的 trace 默认关闭，
可在创建时显式传入 `TraceRecorder`。

## 7. 测试与补充资料

```powershell
.venv\Scripts\python.exe -m pytest -q
```

目前保留 81 个关键测试，覆盖 Agent Loop、三个工具、Schema 校验、连续追问、
Owner/Session 隔离、摘要与快照替换、数据库回滚、请求重试、Run ID 和 trace。
测试使用 Fake LLM 或模拟 SDK，不调用真实 API，不消耗模型额度。

- [单轮运行示例](examples/run_agent.py)
- [领域词汇](CONTEXT.md)
- [AI Prompt 与问题解决记录](docs/AI_PROMPTS.md)

当前实现面向最小可用 Agent：没有向量检索、跨 Session 记忆召回、同 Session 并发写入、
真实搜索或实时天气服务。Owner 隔离仅用于本地数据归属，不替代生产环境的访问控制。

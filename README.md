# mini_agent

从零实现的最小 ReAct 风格 Agent，使用真实 DeepSeek Chat Completions 接口，
支持工具调用、连续追问、独立 Session 和滚动摘要。核心流程不依赖 Agent 框架，
工具 Schema 使用 Pydantic，持久化使用 SQLite。

## 1. 运行方式

需要 Python 3.12+ 和 DeepSeek API Key。在项目根目录执行（PowerShell）：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

编辑 `.env`，填写实际密钥和支持工具调用的模型名称：

```dotenv
API_KEY=your-api-key
BASE_URL=https://api.deepseek.com
MODEL=your-model-name
```

客户端通过 `load_dotenv()` 加载配置，系统环境变量优先于 `.env`。不要提交真实密钥。

```powershell
# 新建对话，可连续输入问题
.venv\Scripts\mini-agent.exe new

# 列出历史 Session，输入编号后继续对话
.venv\Scripts\mini-agent.exe sessions

# 不记录本次会话的执行 trace
.venv\Scripts\mini-agent.exe new --no-trace
```

输入 `/exit` 或在输入时按 `Ctrl+C` 退出。首次非空消息到来时才创建 Session。
本地 Owner ID 自动生成，两个窗口分别 `new` 即得到独立 Session；不支持同时写入同一个 Session。

默认在项目根目录保存 `config.json`、`sessions.db` 和 `traces/<run_id>.json`，运行数据默认被 Git 忽略，
仅下方链接的示例 trace 纳入版本库。
Trace 可能包含对话原文和工具结果，不会自动清理。

## 2. 系统设计

```mermaid
flowchart TD
    CLI["CLI: new / sessions"] --> MANAGER["ConversationManager"]
    MANAGER --> ACTIVE["ActiveConversation<br/>当前 Session 的内存 history"]
    DEFINITION["AgentDefinition<br/>system_prompt + tools"] -->|创建| RUNTIME["AgentRuntime<br/>每次提问新建 RunState"]
    ACTIVE -->|用户输入 + history| RUNTIME
    RUNTIME -->|RunResult| ACTIVE
    DB[("SQLite")] -->|恢复时加载一次| MANAGER
    ACTIVE -->|追加或替换历史| DB
    RUNTIME --> RUNLLM["RunLLM<br/>预算检查 + usage 累计"]
    RUNLLM --> CLIENT["DeepSeekClient"]
    RUNTIME <-->|工具调用与结果| TOOLS["ToolRegistry"]
    RUNTIME -->|调用 LLM 前检查| COMPACT["ContextCompactor"]
    COMPACT -->|compaction 调用| RUNLLM
```

应用复用 Client、Runtime、Registry、Compactor 和 Store；Session 历史只属于各自的
ActiveConversation。每次提问创建新的 RunState、RunLLM 和 Run ID。

`ModelConfig` 控制 chat/compaction 的单次输出（默认 2,048/4,096 tokens）和重试次数；
`RuntimeConfig` 控制 `max_steps=8` 以及 chat/compaction 的 Run 级软预算（默认 50K/800K）。
两类调用分别累计，最终 `token_usage` 为二者之和。

Agent Loop：接收输入 → 调用 LLM → 返回答案，或执行工具并追加结果 → 继续调用 LLM。
使用原生 `content` / `tool_calls` 解析响应，不保存模型内部思考。

每个工具包含名称、描述、Pydantic 参数模型和执行方法，由 Registry 注册并生成 Schema，
LLM 自主选择调用：

- `calculator`：参数 `operation / a / b`，支持加减乘除，除零返回错误。
- `search`：参数 `query`，固定返回 `"mock result"`。
- `weather`：参数 `location`，返回模拟天气，不提供实时数据。

## 3. Memory：召回时机与放置方式

Memory 是当前 Session 的有效历史及可选摘要，不使用向量检索，也不跨 Session 召回。

- **新建对话**：history 为空，不读取历史。
- **恢复 Session**：校验 Owner 后，从 SQLite 加载一次历史到内存。
- **连续追问**：直接使用内存 history，不逐轮读取 SQLite。
- **工具执行后**：将调用与结果追加到当前 Run，供下一次 LLM 调用使用。
- **保存成功后**：更新内存 history，供下一轮提问使用。

每次请求的消息顺序为：

```text
system prompt
历史摘要（如有，作为 assistant 消息）
最近未压缩的完整 user / assistant / tool 轮次
当前用户输入及本轮已产生的 assistant / tool 消息
```

system prompt 仅在请求时注入，不持久化；摘要是历史资料，不是 system 指令。
内部标记 `_kind`、`_run_id` 发送前移除，tools Schema 通过独立的 `tools` 参数传入。
Trace 不参与 memory 召回。

## 4. Context 压缩与保存

每次正常 LLM 调用前，将 messages（包含 system）与 tools Schema 序列化，
以字符数 `/ 4` 估算 token 数。默认按 **1M 上限、70% 触发**；
这只是配置假设，需按模型调整，估算不能保证不超限。

压缩采用“滚动摘要 + 最近完整轮次”：

1. 保留最近 4 轮和当前 Run，工具调用与结果不能拆开，保留轮数不会自动减少。
2. 用同一个 LLM 将“已有摘要 + 更早轮次”合并为一条新摘要。
3. 保留目标、约束、关键事实、工具结果和未完成事项；摘要默认最多 8,000 字符。
4. 检查摘要有效、上下文确实变短且低于预算，再替换运行中的旧历史。

配置由 `AgentDefinition.create_runtime(runtime_config=RuntimeConfig(...),
context_policy=ContextPolicy(...))` 传入。
摘要调用不占用 Agent 的 `max_steps`。没有可压缩旧轮次、预算不足或摘要失败时停止本轮，
不写回历史，也不更新会话内存。

保存规则：

- **未压缩**：只追加本轮 `new_messages`。
- **已压缩**：一个事务内删除该 Session 的全部旧消息，再插入“摘要 + 保留轮次 + 本轮消息”。
- **提交成功后**才更新内存；数据库失败回滚，不影响其他 Session。

SQLite 保存的是有效上下文，不是完整原文档案；被摘要覆盖的细节可能丢失。
Trace 独立保留，不随压缩删除，也不会自动用于找回这些细节。

每个 trace 事件保存当时累计的 `chat_token_usage`、`compaction_token_usage`、
`token_usage` 和 `usage_complete`。模型事件另保存本次请求的 usage；`run.end`、
RunResult 和 CLI 输出最终总量。成功响应缺少 usage 时仍保留输出，但将统计标为不完整。

## AI Prompt 与问题解决记录

- [AI Prompt 与开发思路](docs/AI_PROMPTS.md)：核心设计思路与开发过程中使用的 Prompt。
- [问题解决运行记录（JSON）](traces/9aad315a-ef79-48c9-a9bf-fca4b8d1c11f.json)：
  用户输入 `5+20*1`，Agent 先调用乘法工具得到 `20`，再调用加法工具得到 `25`，
  最终返回答案。记录包含用户 Prompt、assistant 输出、工具参数、执行结果和 Run 状态，
  不包含 system prompt 或模型内部思考。

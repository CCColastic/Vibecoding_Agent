# AI 辅助开发记录

## 开发思路

我先完成最小 Agent Runtime，而不是引入现有 Agent 框架。Runtime 只负责一次
LLM/Tool Loop；AgentDefinition 保存 system prompt 和工具列表，工具通过 Pydantic
Schema 注册，由 LLM 自主决定是否调用。

连续对话由 Session 管理：应用长期复用 Client 和 Runtime，每次提问只创建 RunState。
新对话在第一条消息时创建 Session；恢复对话只从 SQLite 加载一次，后续直接使用内存历史。

上下文接近上限时，以字符数 / 4 粗略估算，在 70% 触发滚动摘要，保留最近完整轮次。
压缩后的有效历史整体替换 SQLite 中该 Session 的旧历史。

最后补充基础可靠性与可观察性：LLM 请求有限重试；每个 Run 使用独立 run_id 和 trace；
ModelConfig 控制单次请求，RuntimeConfig 控制 step 和 Run 级 token 预算，chat 与
compaction 分别累计 usage，并输出总量。

## 核心 Prompt

> 从零实现一个最小可用 Agent，使用真实 DeepSeek Chat Completions，不依赖现有
> Agent 框架；实现 Agent Loop、工具注册、最大步数、异常处理和执行日志。

> 在 Runtime 上层加入 AgentDefinition，接收 system_prompt 和 tools list，
> 负责注册工具并返回可长期复用的 AgentRuntime。

> 支持连续追问。Runtime 不保存 Session 状态；每次提问创建 RunState。
> 新 Session 延迟创建，恢复 Session 时从 SQLite 加载一次，后续追加数据库和内存历史。

> 当上下文达到 1M 的 70% 时进行压缩，使用 context_size / 4 估算。
> 采用滚动摘要并保留最近完整轮次，不实现复杂检索。

> 压缩成功后，用“摘要 + 最近轮次 + 当前 Run”整体替换 SQLite 中该 Session 的历史；
> 删除和重新插入必须在同一个事务中完成。

> 为 llm_call 加入最多三次尝试，只重试连接错误、超时、429 和 5xx，
> 等待时间为 1 秒、2 秒，关闭 SDK 自带重试。

> 为每次执行生成 run_id，记录 user、assistant、tool 和 run.end trace；
> 每个 TraceEvent 记录当前累计 token_usage，最终输出该 Run 的总 usage。

> 分离 ModelConfig 和 RuntimeConfig。chat 与 compaction 使用不同的单次输出限制和
> Run 级预算，由 RunLLM 根据 purpose 切换并累计 usage。

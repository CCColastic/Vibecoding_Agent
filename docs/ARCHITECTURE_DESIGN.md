# Agent Runtime 架构设计题
## 模块一：Context / Performance

### Session 连续聊了 200 轮，context 快满了，如何压缩并保证流畅？

在使用量达到窗口的 60%–70% 时提前触发增量压缩。预留足够空间给下一轮用户输入、模型输出和工具结果。

压缩后的上下文分为四部分：

1. **固定指令**：系统规则、角色设定和安全边界，保持原文，不参与压缩。
2. **会话契约**：当前目标、硬性约束、用户要求、已确认的偏好。
3. **工作状态**：当前计划、完成进度、待办事项、关键决策、失败尝试及其原因。
4. **近期对话**：保留最近 5–10 轮原文，维持语气、指代和局部讨论的连续性。

历史消息不会直接删除，而是保存在外部持久化的 event log 中。压缩摘要中的关键事实、决策和工具结果应关联原始消息 ID，模型需要细节时可以重新检索。

---

## 模块二：Memory

### 用户问了一个半个月前问过的问题，Agent 如何合理召回？

首先要判断当前问题是否需要使用历史记忆，而不是每次都检索全部 memory。

如果需要，我会先使用 Rewriter 对当前问题进行查询改写，补充上下文中的省略信息和实体。

然后进行混合召回，同时使用：

- 向量语义检索；
- BM25 或关键词检索；
- 实体匹配；
- 时间范围；
- 用户、项目和 session scope；
- 记忆的重要性和用户反馈。

召回结果经过合并、去重后，再由 Reranker 根据与当前任务的相关性、时间有效性、来源可信度和用户反馈进行重排与过滤，判断记忆是否过期。最后只向模型注入完成当前回答所需的最小记忆。

完整流程可以概括为：

```text
判断是否需要记忆
→ Query Rewrite
→ 混合召回
→ 合并去重
→ Rerank
→ 时效性与冲突检查
→ 注入最小必要记忆
→ 生成答案
```

---

## 模块三：Task

### 长程任务中模型可能忘记目标，如何解决？

我会把任务目标维护成一个短小、结构化并始终位于上下文顶部的 Goal Capsule，例如：

```yaml
objective:
success_criteria:
constraints:
current_phase:
completed:
next_action:
```

模型每次规划或调用工具之前都需要读取这个对象，并根据成功标准检查当前动作是否仍然服务于原始目标。

这种方案成本低、占用的 token 少，而且能够持续提醒模型当前目标、硬性约束和下一步行动。

它的风险是 Goal Capsule 可能在任务执行过程中被模型错误修改，造成目标漂移。

---

## 模块四：Tool / Session Runtime

### Session busy 时收到新用户消息或异步工具完成事件，Runtime 应如何处理？

我会为每个 session 建立一个串行处理的 mailbox。用户消息、工具完成事件、定时事件和取消请求都先写入 event log，再由 session runtime 按顺序处理，避免多个线程同时修改 session state。

每个事件都要携带：

```text
session_id
run_id
branch_id
event_id
correlation_id
sequence
```

其中 `run_id` 和 `branch_id` 表示事件在时间上和任务空间上属于哪一次执行，`correlation_id` 用来关联原始工具调用。

如果异步工具完成事件属于当前 run：

- 先持久化工具结果；
- 如果模型正在生成，不立即并发启动第二个 Agent loop；
- 将结果标记为 pending；
- 到达安全点后，把工具结果与等待中的用户消息一起交给下一轮模型处理。

如果新用户消息只是补充信息，可以作为 steering input 在安全点注入；如果用户改变目标、要求停止或纠正当前操作，则应拥有更高优先级，可以取消当前生成或阻止后续工具执行。

---

## 模块五：Agent Runtime 架构对比

### Claude Code 的工具输出与 OpenAI-compatible function calling 有什么不同？

Anthropic 使用的是 content block 结构。Assistant 消息中可以同时包含文本和 `tool_use` block，工具执行结果则以 `tool_result` block 放在下一条 user 消息中，通过 `tool_use_id` 与调用关联。

```text
assistant:
  text
  tool_use

user:
  tool_result
  text
```

它没有单独的 `tool` 或 `function` role，而是把工具调用和结果直接融入 user/assistant 消息。`tool_result.content` 还可以包含文本、图片等不同类型的内容块，因此更适合多模态结果和 Agent 的流式执行过程。

OpenAI-compatible Chat Completions 通常采用：

```text
assistant.tool_calls
tool:
  tool_call_id
  content
```

它使用独立的 `tool` role 返回结果，协议结构更加标准化，拥有较好的 SDK、网关和模型兼容性，因此国内模型很容易通过兼容这一接口降低接入成本。

两者的取舍是：

- Anthropic content block 表达能力更强，文本、工具调用和多模态结果可以自然组合，但消息顺序和 block 解析要求更严格。
- OpenAI-compatible 接入简单、生态成熟，但许多兼容实现只支持基础的 JSON 参数和文本工具结果，对并行调用、多模态结果、增量输出和错误语义的支持并不完全一致。

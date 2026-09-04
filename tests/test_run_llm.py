import logging

import pytest

from mini_agent.llm import LLMResponse, LLMUsage
from mini_agent.llm.run import RunLLM, RunUsage, TokenBudgetExceeded
from mini_agent.runtime_config import RuntimeConfig


class QueueClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def llm_call(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def answer(tokens=None):
    usage = None if tokens is None else LLMUsage(tokens - 1, 1, tokens)
    return LLMResponse({"content": "ok"}, usage, "stop")


async def test_run_llm_enforces_independent_soft_budgets_and_totals():
    client = QueueClient([answer(60), answer(60), answer(500)])
    usage = RunUsage()
    calls = RunLLM(
        client=client,
        config=RuntimeConfig(max_chat_usage=100, max_compaction_usage=500),
        usage=usage,
    )
    await calls.call(purpose="chat", messages=[], tools=[])
    await calls.call(purpose="chat", messages=[], tools=[])
    with pytest.raises(TokenBudgetExceeded, match="chat.*120.*100"):
        await calls.call(purpose="chat", messages=[], tools=[])
    await calls.call(purpose="compaction", messages=[], tools=[])
    with pytest.raises(TokenBudgetExceeded, match="compaction.*500.*500"):
        await calls.call(purpose="compaction", messages=[], tools=[])
    assert usage.chat_tokens == 120
    assert usage.compaction_tokens == 500
    assert usage.total_tokens == 620
    assert usage.complete
    assert [call["purpose"] for call in client.calls] == [
        "chat", "chat", "compaction",
    ]


async def test_run_llm_keeps_missing_usage_output_and_marks_total_incomplete(caplog):
    client = QueueClient([answer(None), answer(10)])
    usage = RunUsage()
    calls = RunLLM(client=client, config=RuntimeConfig(), usage=usage)
    with caplog.at_level(logging.WARNING):
        first = await calls.call(purpose="chat", messages=[], tools=[])
    second = await calls.call(purpose="chat", messages=[], tools=[])
    assert first.message["content"] == second.message["content"] == "ok"
    assert usage.chat_tokens == usage.total_tokens == 10
    assert not usage.complete
    assert "usage unavailable" in caplog.text.lower()


async def test_run_llm_marks_usage_incomplete_when_request_raises():
    client = QueueClient([RuntimeError("provider failure")])
    usage = RunUsage()
    calls = RunLLM(client=client, config=RuntimeConfig(), usage=usage)
    with pytest.raises(RuntimeError, match="provider failure"):
        await calls.call(purpose="chat", messages=[], tools=[])
    assert not usage.complete
    assert usage.total_tokens == 0

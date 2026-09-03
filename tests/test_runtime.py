import json

from mini_agent import AgentDefinition
from mini_agent.tools import CalculatorTool, SearchTool, WeatherTool
from tests.fakes import FailingLLMClient, FakeLLMClient, message_payloads, tool_call


def make_definition() -> AgentDefinition:
    return AgentDefinition(
        "Use tools when useful",
        [CalculatorTool(), SearchTool(), WeatherTool()],
    )


async def test_runtime_returns_direct_answer() -> None:
    client = FakeLLMClient([{"role": "assistant", "content": "Hello"}])
    runtime = AgentDefinition("Use tools when useful", []).create_runtime(llm_client=client)

    result = await runtime.run("Hi")

    assert result.status == "completed"
    assert result.final_answer == "Hello"
    assert result.steps_used == 1
    assert result.tool_executions == []
    assert client.calls[0]["tools"] == []
    assert message_payloads(result.new_messages) == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    assert client.calls[0]["messages"] == [
        {"role": "system", "content": "Use tools when useful"},
        {"role": "user", "content": "Hi"},
    ]


async def test_runtime_executes_multiple_tool_calls_in_order() -> None:
    client = FakeLLMClient(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    tool_call("search", '{"query":"news"}', call_id="search-1"),
                    tool_call(
                        "weather",
                        '{"location":"Hong Kong"}',
                        call_id="weather-1",
                    ),
                ],
            },
            {"role": "assistant", "content": "Done"},
        ]
    )
    runtime = make_definition().create_runtime(llm_client=client)

    result = await runtime.run("Search and check weather")

    assert [execution.name for execution in result.tool_executions] == [
        "search",
        "weather",
    ]
    assert [message["tool_call_id"] for message in result.messages if message["role"] == "tool"] == [
        "search-1",
        "weather-1",
    ]


async def test_invalid_tool_arguments_are_returned_to_model() -> None:
    client = FakeLLMClient(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call("calculator", '{"operation":"add","a":1}')],
            },
            {"role": "assistant", "content": "I could not calculate that."},
        ]
    )
    runtime = make_definition().create_runtime(llm_client=client)

    result = await runtime.run("Calculate")

    assert result.status == "completed"
    assert result.tool_executions[0].result.ok is False
    assert "Invalid arguments" in result.tool_executions[0].result.content
    assert json.loads(client.calls[1]["messages"][-1]["content"])["ok"] is False


async def test_runtime_stops_at_max_steps() -> None:
    client = FakeLLMClient(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call("search", '{"query":"one"}', call_id="1")],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call("search", '{"query":"two"}', call_id="2")],
            },
        ]
    )
    runtime = make_definition().create_runtime(max_steps=2, llm_client=client)

    result = await runtime.run("Keep searching")

    assert result.status == "max_steps_exceeded"
    assert result.steps_used == 2
    assert len(client.calls) == 2


async def test_empty_llm_response_is_protocol_error() -> None:
    client = FakeLLMClient([{"role": "assistant", "content": None}])
    runtime = make_definition().create_runtime(llm_client=client)

    result = await runtime.run("Hi")

    assert result.status == "llm_protocol_error"
    assert result.steps_used == 1


async def test_llm_exception_returns_basic_safe_error() -> None:
    runtime = make_definition().create_runtime(llm_client=FailingLLMClient())

    result = await runtime.run("Hi")

    assert result.status == "llm_error"
    assert result.error == "LLM request failed: RuntimeError"
    assert "secret" not in result.error

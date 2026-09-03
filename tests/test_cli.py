from mini_agent.app import build_conversation_manager
from mini_agent.cli import run_cli
from tests.fakes import FakeLLMClient


def input_from(*values: str):
    iterator = iter(values)
    return lambda prompt: next(iterator)


async def test_new_command_supports_multiple_turns(tmp_path) -> None:
    client = FakeLLMClient(
        [
            {"role": "assistant", "content": "Hello Ada"},
            {"role": "assistant", "content": "Your name is Ada"},
        ]
    )
    conversations = build_conversation_manager(data_dir=tmp_path, llm_client=client)
    output: list[str] = []

    status = await run_cli(
        ["new"],
        conversation_manager=conversations,
        input_fn=input_from("My name is Ada", "What is my name?", "/exit"),
        output_fn=output.append,
    )

    sessions = conversations.list_sessions()
    assert status == 0
    assert len(sessions) == 1
    assert sessions[0].title == "My name is Ada"
    assert client.calls[1]["messages"][1:3] == [
        {"role": "user", "content": "My name is Ada"},
        {"role": "assistant", "content": "Hello Ada"},
    ]
    assert any(line.startswith("Session: ") for line in output)
    assert output[-1] == "Agent: Your name is Ada"


async def test_new_command_does_not_create_empty_session(tmp_path) -> None:
    conversations = build_conversation_manager(
        data_dir=tmp_path, llm_client=FakeLLMClient([])
    )

    status = await run_cli(
        ["new"],
        conversation_manager=conversations,
        input_fn=input_from("/exit"),
        output_fn=lambda message: None,
    )

    assert status == 0
    assert conversations.list_sessions() == []


async def test_sessions_command_resumes_selected_conversation(tmp_path) -> None:
    client = FakeLLMClient(
        [
            {"role": "assistant", "content": "Saved answer"},
            {"role": "assistant", "content": "Follow-up answer"},
        ]
    )
    conversations = build_conversation_manager(data_dir=tmp_path, llm_client=client)
    conversation = conversations.new_conversation()
    await conversation.send_message("Original question")
    output: list[str] = []

    status = await run_cli(
        ["sessions"],
        conversation_manager=conversations,
        input_fn=input_from("1", "Follow-up", "/exit"),
        output_fn=output.append,
    )

    assert status == 0
    assert client.calls[1]["messages"][1:3] == [
        {"role": "user", "content": "Original question"},
        {"role": "assistant", "content": "Saved answer"},
    ]
    assert output[0].startswith("[1] Original question")
    assert output[-1] == "Agent: Follow-up answer"


async def test_sessions_command_handles_empty_list(tmp_path) -> None:
    conversations = build_conversation_manager(
        data_dir=tmp_path, llm_client=FakeLLMClient([])
    )
    output: list[str] = []

    status = await run_cli(
        ["sessions"], conversation_manager=conversations, output_fn=output.append
    )

    assert status == 0
    assert output == ["No sessions found."]


async def test_sessions_command_rejects_invalid_selection(tmp_path) -> None:
    client = FakeLLMClient([{"role": "assistant", "content": "Answer"}])
    conversations = build_conversation_manager(data_dir=tmp_path, llm_client=client)
    conversation = conversations.new_conversation()
    await conversation.send_message("Question")
    output: list[str] = []

    status = await run_cli(
        ["sessions"],
        conversation_manager=conversations,
        input_fn=input_from("99"),
        output_fn=output.append,
    )

    assert status == 1
    assert output[-1] == "Invalid session selection."

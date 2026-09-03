from mini_agent.app import build_application
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
    app = build_application(data_dir=tmp_path, llm_client=client)
    output: list[str] = []

    status = await run_cli(
        ["new"],
        application=app,
        input_fn=input_from("My name is Ada", "What is my name?", "/exit"),
        output_fn=output.append,
    )

    sessions = app.session_service.list_sessions(app.owner.owner_id)
    stored = app.session_store.load_messages(app.owner.owner_id, sessions[0].id)
    assert status == 0
    assert len(sessions) == 1
    assert sessions[0].title == "My name is Ada"
    assert len(stored) == 4
    assert client.calls[1]["messages"][1:3] == [
        {"role": "user", "content": "My name is Ada"},
        {"role": "assistant", "content": "Hello Ada"},
    ]
    assert any(line.startswith("Session: ") for line in output)
    assert output[-1] == "Agent: Your name is Ada"


async def test_new_command_does_not_create_empty_session(tmp_path) -> None:
    app = build_application(data_dir=tmp_path, llm_client=FakeLLMClient([]))

    status = await run_cli(
        ["new"],
        application=app,
        input_fn=input_from("/exit"),
        output_fn=lambda message: None,
    )

    assert status == 0
    assert app.session_service.list_sessions(app.owner.owner_id) == []


async def test_sessions_command_resumes_selected_conversation(tmp_path) -> None:
    client = FakeLLMClient(
        [
            {"role": "assistant", "content": "Saved answer"},
            {"role": "assistant", "content": "Follow-up answer"},
        ]
    )
    app = build_application(data_dir=tmp_path, llm_client=client)
    conversation = app.session_service.new_conversation(app.owner.owner_id)
    await conversation.send_message("Original question")
    output: list[str] = []

    status = await run_cli(
        ["sessions"],
        application=app,
        input_fn=input_from("1", "Follow-up", "/exit"),
        output_fn=output.append,
    )

    session = app.session_service.list_sessions(app.owner.owner_id)[0]
    stored = app.session_store.load_messages(app.owner.owner_id, session.id)
    assert status == 0
    assert len(stored) == 4
    assert client.calls[1]["messages"][1:3] == [
        {"role": "user", "content": "Original question"},
        {"role": "assistant", "content": "Saved answer"},
    ]
    assert output[0].startswith("[1] Original question")
    assert output[-1] == "Agent: Follow-up answer"


async def test_sessions_command_handles_empty_list(tmp_path) -> None:
    app = build_application(data_dir=tmp_path, llm_client=FakeLLMClient([]))
    output: list[str] = []

    status = await run_cli(
        ["sessions"], application=app, output_fn=output.append
    )

    assert status == 0
    assert output == ["No sessions found."]


async def test_sessions_command_rejects_invalid_selection(tmp_path) -> None:
    client = FakeLLMClient([{"role": "assistant", "content": "Answer"}])
    app = build_application(data_dir=tmp_path, llm_client=client)
    conversation = app.session_service.new_conversation(app.owner.owner_id)
    await conversation.send_message("Question")
    output: list[str] = []

    status = await run_cli(
        ["sessions"],
        application=app,
        input_fn=input_from("99"),
        output_fn=output.append,
    )

    assert status == 1
    assert output[-1] == "Invalid session selection."

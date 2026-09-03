import json

import mini_agent.app as app
from mini_agent.app import build_conversation_manager
from tests.fakes import FakeLLMClient


async def test_builder_returns_working_owner_scoped_manager(tmp_path) -> None:
    client = FakeLLMClient(
        [
            {"role": "assistant", "content": "First answer"},
            {"role": "assistant", "content": "Second answer"},
            {"role": "assistant", "content": "Follow-up answer"},
        ]
    )

    conversations = build_conversation_manager(data_dir=tmp_path, llm_client=client)
    first = conversations.new_conversation()
    second = conversations.new_conversation()

    first_result = await first.send_message("First question")
    await second.send_message("Second question")
    resumed = conversations.resume_conversation(first.session_id)
    result = await resumed.send_message("Follow-up")

    assert result.final_answer == "Follow-up answer"
    assert len(conversations.list_sessions()) == 2
    trace = json.loads((tmp_path / "traces" / f"{first_result.run_id}.json").read_text(encoding="utf-8"))
    assert all(event["session_id"] == first.session_id for event in trace)
    assert len(client.calls[1]["messages"]) == 2
    assert client.calls[2]["messages"][1:3] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]


async def test_default_builder_keeps_all_generated_data_in_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    monkeypatch.setattr(app, "__file__", str(project / "src" / "mini_agent" / "app.py"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINI_AGENT_DATA_DIR", str(tmp_path / "old-override"))
    manager = build_conversation_manager(llm_client=FakeLLMClient([{"content": "Answer"}]))
    result = await manager.new_conversation().send_message("Hello")
    assert (project / "config.json").is_file()
    assert (project / "sessions.db").is_file()
    assert (project / "traces" / f"{result.run_id}.json").is_file()
    assert not (tmp_path / "config.json").exists()
    assert not (tmp_path / "old-override").exists()

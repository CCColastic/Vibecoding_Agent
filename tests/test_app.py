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

    await first.send_message("First question")
    await second.send_message("Second question")
    resumed = conversations.resume_conversation(first.session_id)
    result = await resumed.send_message("Follow-up")

    assert result.final_answer == "Follow-up answer"
    assert len(conversations.list_sessions()) == 2
    assert client.calls[2]["messages"][1:3] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]

from mini_agent.app import build_application
from tests.fakes import FakeLLMClient


def test_application_builds_one_shared_runtime_and_client(tmp_path) -> None:
    client = FakeLLMClient([])

    app = build_application(data_dir=tmp_path, llm_client=client)
    first = app.session_service.new_conversation(app.owner.owner_id)
    second = app.session_service.new_conversation(app.owner.owner_id)

    assert app.llm_client is client
    assert app.runtime.llm_client is client
    assert first.runtime is app.runtime
    assert second.runtime is app.runtime
    assert first.session_store is app.session_store
    assert second.session_store is app.session_store

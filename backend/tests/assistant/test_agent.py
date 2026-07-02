import json

import pytest
from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart, models
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.assistant.agent import build_document_agent_model, chat_model_name, document_agent, load_instructions
from app.assistant.deps import DocumentAgentDeps
from app.assistant.outputs import GroundedAnswer
from tests.assistant.test_deps import StubRetriever, StubValidator, THREAD_ID, USER_ID

models.ALLOW_MODEL_REQUESTS = False


def test_load_instructions_encodes_grounding_contract() -> None:
    instructions = load_instructions()

    assert "GroundedAnswer" in instructions
    assert "search_filings" in instructions
    assert "read_chunk" in instructions
    assert "read_surrounding_chunks" in instructions
    assert "stock recommendation" in instructions.lower()


def test_document_agent_registers_retrieval_tools() -> None:
    tool_names = sorted(document_agent._function_toolset.tools.keys())

    assert tool_names == ["read_chunk", "read_surrounding_chunks", "search_filings"]


def test_chat_model_name_uses_google_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.assistant.agent.settings.google_api_key", "test-google-key")

    assert chat_model_name("google", "gemini-2.0-flash") == "google:gemini-2.0-flash"

    model = build_document_agent_model("google", "gemini-2.0-flash")
    assert model.model_name == "gemini-2.0-flash"


def test_chat_model_name_uses_opencode_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.assistant.agent.settings.opencode_api_key", "test-opencode-key")

    assert chat_model_name("opencode", "glm-5.2") == "opencode:glm-5.2"

    model = build_document_agent_model("opencode", "glm-5.2")
    assert model.model_name == "glm-5.2"


@pytest.mark.anyio
async def test_document_agent_run_invokes_search_filings_tool() -> None:
    retriever = StubRetriever()
    deps = DocumentAgentDeps(
        user_id=USER_ID,
        thread_id=THREAD_ID,
        retriever=retriever,
        grounding_validator=StubValidator(),
    )
    step = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal step
        step += 1
        if step == 1:
            return ModelResponse(
                parts=[ToolCallPart("search_filings", {"query": "AWS operating income"})]
            )

        return ModelResponse(
            parts=[
                TextPart(
                    json.dumps(
                        {
                            "answer": "This corpus does not contain enough evidence to answer that.",
                            "citations": [],
                            "cited_passages": [],
                        }
                    )
                )
            ]
        )

    with document_agent.override(model=FunctionModel(model_fn)):
        result = await document_agent.run("How did AWS operating income change?", deps=deps)

    assert retriever.last_query == "AWS operating income"
    assert result.output == GroundedAnswer(
        answer="This corpus does not contain enough evidence to answer that.",
        citations=[],
        cited_passages=[],
    )

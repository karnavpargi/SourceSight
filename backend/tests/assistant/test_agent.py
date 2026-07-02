from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart, models
from pydantic_ai.models.function import AgentInfo, FunctionModel
import pytest

from app.assistant.agent import document_agent, load_instructions
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
                ToolCallPart(
                    "final_result",
                    {
                        "answer": "This corpus does not contain enough evidence to answer that.",
                        "citations": [],
                        "cited_passages": [],
                    },
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

import os
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console

from app import (
    GraphState,
    decide_to_generate,
    generate,
    grade_documents,
    grade_generation,
    normal_llm,
    retrieve,
    route_question,
    transform_query,
    workflow,
)

os.environ["OPENAI_API_KEY"] = "dummy-api-key-for-testing"


@pytest.fixture
def mock_state():
    return GraphState(
        question="What is the power consumption of Osram LED bulbs?",
        generation="",
        documents=[],
        retries=0,
    )


@pytest.fixture
def mock_documents():
    return [
        Document(
            page_content="Osram LED bulbs consume 5-10 watts of power.",
            metadata={"source": "doc1"},
        ),
        Document(
            page_content="Osram offers a wide range of LED bulbs.",
            metadata={"source": "doc2"},
        ),
    ]


def test_retrieve(mock_state, mock_documents):
    with patch("app.ensemble_retriever.invoke", return_value=mock_documents):
        result = retrieve(mock_state)

    assert len(result["documents"]) == 2
    assert result["question"] == mock_state["question"]


def test_generate(mock_state, mock_documents):
    mock_state["documents"] = mock_documents
    with patch(
        "app.rag_chain",
        return_value="Osram LED bulbs typically consume 5-10 watts of power.",
    ):
        result = generate(mock_state)

    assert "generation" in result
    assert result["generation"].startswith("Osram LED bulbs")


def test_grade_documents(mock_state, mock_documents):
    mock_state["documents"] = mock_documents
    with patch("app.retrieval_grader", return_value={"score": "yes"}):
        result = grade_documents(mock_state)

    assert len(result["documents"]) == 2


def test_transform_query(mock_state):
    with patch(
        "app.question_rewriter.run",
        return_value="What is the power consumption range of Osram LED light bulbs?",
    ):
        result = transform_query(mock_state)

    assert result["question"] != mock_state["question"]
    assert "power consumption" in result["question"]


def test_normal_llm(mock_state):
    with patch(
        "app.answer_normal.run",
        return_value="Osram LED bulbs are energy-efficient lighting solutions.",
    ):
        result = normal_llm(mock_state)

    assert "generation" in result
    assert "Osram LED bulbs" in result["generation"]


def test_route_question(mock_state):
    with patch("app.question_router", return_value={"datasource": "vectorstore"}):
        result = route_question(mock_state)

    assert result == "vectorstore"


def test_decide_to_generate(mock_state, mock_documents):
    from app import console  # Import console here

    result = decide_to_generate(mock_state)
    assert result == "transform_query"

    mock_state["documents"] = mock_documents
    result = decide_to_generate(mock_state)
    assert result == "generate"


def test_grade_generation(mock_state, mock_documents):
    mock_state["documents"] = mock_documents
    mock_state["generation"] = "Osram LED bulbs typically consume 5-10 watts of power."

    with patch(
        "app.hallucination_grader", return_value={"score": "useful"}
    ):  # Ensure mock returns correct value
        result = grade_generation(mock_state)

    assert result == "useful"  # Make sure the expected value is returned


@pytest.mark.asyncio
async def test_workflow_integration():
    with patch(
        "app.ensemble_retriever.get_relevant_documents",
        return_value=[
            Document(
                page_content="Osram LED bulbs consume 5-10 watts of power.",
                metadata={"source": "doc1"},
            )
        ],
    ), patch(
        "app.rag_chain",
        return_value="Osram LED bulbs typically consume 5-10 watts of power.",
    ), patch(
        "app.retrieval_grader", return_value={"score": "yes"}
    ), patch(
        "app.hallucination_grader", return_value={"score": "yes"}
    ):

        inputs = {"question": "What is the power consumption of Osram LED bulbs?"}
        result = None
        async for output in workflow.astream(inputs):
            for key, value in output.items():
                if "generation" in value:
                    result = value["generation"]

        assert result is not None
        assert "watts" in result


if __name__ == "__main__":
    pytest.main()

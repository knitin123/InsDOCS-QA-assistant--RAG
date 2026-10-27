from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import modules.load_vectorstore as vs
from modules.query_handlers import NO_CONTEXT_ANSWER, query_chain


def test_query_chain_passes_retrieved_context_to_llm():
    seen = {}

    def fake_chain(inputs):
        seen.update(inputs)
        return "answer"

    docs = [
        Document(page_content="Insulin regulates glucose.", metadata={"source": "/x/DIABETES.pdf"}),
        Document(page_content="Type 2 is most common.", metadata={"source": "/x/DIABETES.pdf"}),
    ]

    result = query_chain(RunnableLambda(fake_chain), "What is insulin?", docs)

    assert result == {"response": "answer", "sources": ["DIABETES.pdf"]}
    assert "Insulin regulates glucose." in seen["context"]
    assert "Type 2 is most common." in seen["context"]
    assert seen["question"] == "What is insulin?"


def test_query_chain_without_context_skips_llm():
    def must_not_run(_):
        raise AssertionError("LLM should not be called without context")

    result = query_chain(RunnableLambda(must_not_run), "anything", [])

    assert result == {"response": NO_CONTEXT_ANSWER, "sources": []}


def test_retrieve_returns_documents_from_index(fake_index, fake_embeddings):
    fake_index.upsert([
        ("doc-0", fake_embeddings.embed_query("insulin"), {"text": "Insulin chunk", "source": "a.pdf"}),
        ("doc-1", fake_embeddings.embed_query("unrelated"), {"text": "Other chunk", "source": "b.pdf"}),
    ])

    docs = vs.retrieve("insulin", top_k=1)

    assert [d.page_content for d in docs] == ["Insulin chunk"]
    assert docs[0].metadata["source"] == "a.pdf"

"""Shared fixtures: in-memory stand-ins for Pinecone, Google embeddings and Groq,
so the whole RAG pipeline runs offline without API keys."""
import os
from pathlib import Path

import pytest

# dummy keys so client constructors don't complain; nothing reaches the network
for key in ("GOOGLE_API_KEY", "PINECONE_API_KEY", "GROQ_API_KEY"):
    os.environ.setdefault(key, "test-key")

from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import modules.llm as llm_module
import modules.load_vectorstore as vs
import modules.pdf_handlers as pdf_handlers

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
SAMPLE_PDF = ASSETS_DIR / "DIABETES.pdf"


class FakeIndex:
    """Minimal in-memory version of a Pinecone index (upsert + dot-product query)."""

    def __init__(self):
        self.vectors = {}
        self.upsert_calls = 0

    def upsert(self, vectors):
        self.upsert_calls += 1
        for vec_id, values, metadata in vectors:
            self.vectors[vec_id] = (values, metadata)

    def query(self, vector, top_k, include_metadata=True):
        scored = sorted(
            ((sum(a * b for a, b in zip(vector, values)), vec_id, metadata)
             for vec_id, (values, metadata) in self.vectors.items()),
            reverse=True,
        )
        return {"matches": [
            {"id": vec_id, "score": score, "metadata": metadata}
            for score, vec_id, metadata in scored[:top_k]
        ]}


@pytest.fixture
def fake_index(monkeypatch):
    index = FakeIndex()
    monkeypatch.setattr(vs, "get_index", lambda: index)
    return index


@pytest.fixture
def fake_embeddings(monkeypatch):
    embeddings = DeterministicFakeEmbedding(size=vs.EMBED_DIM)
    monkeypatch.setattr(vs, "get_embed_model", lambda: embeddings)
    return embeddings


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeListChatModel(responses=["Diabetes is a chronic condition affecting blood sugar."])
    monkeypatch.setattr(llm_module, "get_llm", lambda: llm)
    return llm


@pytest.fixture(autouse=True)
def tmp_upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_handlers, "UPLOAD_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def client():
    from main import app
    return TestClient(app)

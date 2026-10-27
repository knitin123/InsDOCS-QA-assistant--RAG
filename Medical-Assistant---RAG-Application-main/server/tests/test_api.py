"""End-to-end through the FastAPI app: upload a PDF, then ask about it."""
from conftest import SAMPLE_PDF


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_then_ask(client, fake_index, fake_embeddings, fake_llm):
    with SAMPLE_PDF.open("rb") as f:
        upload = client.post("/upload_pdfs/", files=[("files", ("DIABETES.pdf", f, "application/pdf"))])

    assert upload.status_code == 200
    assert upload.json()["chunks"] > 0

    answer = client.post("/ask/", data={"question": "What is diabetes?"})

    assert answer.status_code == 200
    body = answer.json()
    assert body["response"] == "Diabetes is a chronic condition affecting blood sugar."
    assert body["sources"] == ["DIABETES.pdf"]


def test_ask_with_empty_index_returns_fallback(client, fake_index, fake_embeddings, fake_llm):
    answer = client.post("/ask/", data={"question": "What is diabetes?"})

    assert answer.status_code == 200
    assert "couldn't find relevant information" in answer.json()["response"]
    assert answer.json()["sources"] == []


def test_ask_requires_question(client):
    assert client.post("/ask/", data={}).status_code == 422


def test_ask_reports_backend_failure_as_500(client, fake_embeddings, monkeypatch):
    import modules.load_vectorstore as vs

    def broken_index():
        raise RuntimeError("pinecone down")

    monkeypatch.setattr(vs, "get_index", broken_index)

    answer = client.post("/ask/", data={"question": "hi"})

    assert answer.status_code == 500
    assert answer.json() == {"error": "pinecone down"}

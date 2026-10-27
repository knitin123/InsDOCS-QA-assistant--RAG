"""Live smoke test against real Pinecone, Google embeddings and Groq.

Runs only with `pytest -m integration` and real keys in the environment
(in CI they come from repository secrets)."""
import os
import time

import pytest
from fastapi.testclient import TestClient

from conftest import SAMPLE_PDF

REQUIRED_KEYS = ("GOOGLE_API_KEY", "PINECONE_API_KEY", "GROQ_API_KEY")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        # unset GitHub secrets arrive as empty strings; conftest fills in "test-key" otherwise
        any(os.getenv(k, "") in ("", "test-key") for k in REQUIRED_KEYS),
        reason="needs real GOOGLE_API_KEY, PINECONE_API_KEY and GROQ_API_KEY",
    ),
]


def test_live_rag_pipeline():
    from main import app
    client = TestClient(app)

    with SAMPLE_PDF.open("rb") as f:
        upload = client.post("/upload_pdfs/", files=[("files", ("DIABETES.pdf", f, "application/pdf"))])
    assert upload.status_code == 200, upload.text

    # Pinecone is eventually consistent; give fresh vectors a moment to become queryable
    for _ in range(10):
        answer = client.post("/ask/", data={"question": "What is diabetes?"})
        assert answer.status_code == 200, answer.text
        if answer.json()["sources"]:
            break
        time.sleep(3)

    body = answer.json()
    assert "DIABETES.pdf" in body["sources"]
    assert "diabet" in body["response"].lower()

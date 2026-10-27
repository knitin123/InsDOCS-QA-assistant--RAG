import io

from fastapi import UploadFile

import modules.load_vectorstore as vs
from conftest import SAMPLE_PDF


def test_chunk_pdf_splits_into_bounded_chunks():
    chunks = vs.chunk_pdf(str(SAMPLE_PDF))

    assert len(chunks) > 1
    assert all(len(c.page_content) <= vs.CHUNK_SIZE for c in chunks)
    assert all(c.metadata["source"].endswith("DIABETES.pdf") for c in chunks)


def test_load_vectorstore_stores_chunk_text_in_metadata(fake_index, fake_embeddings, tmp_upload_dir):
    upload = UploadFile(file=io.BytesIO(SAMPLE_PDF.read_bytes()), filename="DIABETES.pdf")

    count = vs.load_vectorstore([upload])

    assert count == len(fake_index.vectors) > 0
    assert (tmp_upload_dir / "DIABETES.pdf").exists()
    values, metadata = fake_index.vectors["DIABETES-0"]
    assert len(values) == vs.EMBED_DIM
    # the query side reads the chunk back from metadata["text"]
    assert metadata["text"].strip()


def test_load_vectorstore_upserts_in_batches(fake_index, fake_embeddings, monkeypatch):
    monkeypatch.setattr(vs, "UPSERT_BATCH_SIZE", 5)
    upload = UploadFile(file=io.BytesIO(SAMPLE_PDF.read_bytes()), filename="DIABETES.pdf")

    count = vs.load_vectorstore([upload])

    assert fake_index.upsert_calls == -(-count // 5)


def test_metadata_is_pinecone_compatible():
    clean = vs.clean_metadata({"source": "a.pdf", "page": 1, "ok": True, "missing": None, "nested": {"x": 1}})

    assert clean == {"source": "a.pdf", "page": 1, "ok": True}


def test_upload_filename_cannot_escape_upload_dir(tmp_upload_dir):
    from modules.pdf_handlers import save_uploaded_files

    upload = UploadFile(file=io.BytesIO(b"%PDF"), filename="../../evil.pdf")
    [path] = save_uploaded_files([upload])

    assert path == str(tmp_upload_dir / "evil.pdf")

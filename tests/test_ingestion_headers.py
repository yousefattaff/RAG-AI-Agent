import os

os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")

from langchain_core.documents import Document

from src.ingestion import chunk_documents, compute_chunk_id


def test_chunk_documents_adds_seq_index_and_header():
    pages = [
        Document(
            page_content="Alpha paragraph about diabetes treatment options and insulin therapy. " * 20,
            metadata={"source": "paper.pdf", "page": 0},
        ),
        Document(
            page_content="Beta paragraph about blood glucose monitoring devices and HbA1c. " * 20,
            metadata={"source": "paper.pdf", "page": 1},
        ),
    ]
    chunks = chunk_documents(pages)
    assert chunks
    assert all("seq_index" in c.metadata for c in chunks)
    assert chunks[0].metadata["seq_index"] == 0
    assert chunks[-1].metadata["seq_index"] == len(chunks) - 1
    assert chunks[0].page_content.startswith("[Document: paper.pdf | Page:")


def test_compute_chunk_id_is_stable_for_same_content():
    doc = Document(
        page_content="hello",
        metadata={"source": "a.pdf", "page": 2},
    )
    assert compute_chunk_id(doc) == compute_chunk_id(doc)
    assert compute_chunk_id(doc).startswith("a.pdf:2:")

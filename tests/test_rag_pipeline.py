import os

os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")

from langchain_core.documents import Document

from src.rag_utils import (
    add_contextual_header,
    compress_chunk,
    document_key,
    expand_with_neighbors,
    reciprocal_rank_fusion,
    unique_documents,
)


def test_contextual_header_prefixes_filename_and_page():
    doc = Document(
        page_content="Insulin lowers blood glucose.",
        metadata={"source": r"C:\papers\study.pdf", "page": 3},
    )
    headed = add_contextual_header(doc)
    assert headed.page_content.startswith("[Document: study.pdf | Page: 3]")
    assert "Insulin lowers blood glucose." in headed.page_content


def test_document_key_uses_source_page_and_index():
    doc = Document(
        page_content="x",
        metadata={"source": "a.pdf", "page": 1, "seq_index": 4},
    )
    assert document_key(doc) == ("a.pdf", 1, 4)


def test_unique_documents_keeps_first_occurrence():
    a = Document(page_content="same", metadata={"source": "a.pdf", "page": 1, "seq_index": 0})
    b = Document(page_content="same", metadata={"source": "a.pdf", "page": 1, "seq_index": 0})
    c = Document(page_content="other", metadata={"source": "a.pdf", "page": 2, "seq_index": 1})
    result = unique_documents([a, b, c])
    assert len(result) == 2
    assert result[0] is a
    assert result[1] is c


def test_reciprocal_rank_fusion_promotes_docs_in_both_lists():
    shared = Document(page_content="both", metadata={"source": "s.pdf", "page": 1, "seq_index": 0})
    only_vec = Document(page_content="vec", metadata={"source": "s.pdf", "page": 1, "seq_index": 1})
    only_kw = Document(page_content="kw", metadata={"source": "s.pdf", "page": 1, "seq_index": 2})

    fused = reciprocal_rank_fusion(
        [[only_vec, shared], [only_kw, shared]],
        k=60,
    )
    assert fused[0].page_content == "both"


def test_expand_with_neighbors_concatenates_same_file_sequence():
    corpus = [
        Document(page_content="A", metadata={"source": "s.pdf", "page": 1, "seq_index": 0}),
        Document(page_content="B", metadata={"source": "s.pdf", "page": 1, "seq_index": 1}),
        Document(page_content="C", metadata={"source": "s.pdf", "page": 2, "seq_index": 2}),
    ]
    hit = corpus[1]
    expanded = expand_with_neighbors([hit], corpus, num_neighbors=1)
    assert len(expanded) == 1
    assert expanded[0].page_content == "A\n\nB\n\nC"
    assert expanded[0].metadata["seq_index"] == 1


def test_compress_chunk_keeps_sentences_overlapping_the_query():
    text = (
        "The capital of France is Paris. "
        "Insulin is a peptide hormone. "
        "Paris is known for the Eiffel Tower."
    )
    compressed = compress_chunk("what is insulin", text)
    assert "Insulin is a peptide hormone." in compressed
    assert "Eiffel Tower" not in compressed

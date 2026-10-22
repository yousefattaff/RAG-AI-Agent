# src/ingestion.py
import sys
import hashlib
import logging
from functools import lru_cache
from pathlib import Path
import chromadb
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import settings, VECTORSTORE_DIR


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
logger = logging.getLogger(__name__)






def _embeddings_for(model_name: str, device: str) -> HuggingFaceEmbeddings:
    """Instantiates an embedding model for the given name/device (cached)."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=4)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns the local HuggingFace embedding model.
    First call downloads ~80MB model. Subsequent calls use cache.

    The result is cached so ingestion, retrieval, and reopening a collection
    all share one model instance instead of reloading it on every call.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _embeddings_for(settings.embedding_model, device)


def load_pdf(pdf_path: Path) -> list:
    """
    Loads a PDF and returns a list of Document objects, one per page.

    Each Document has:
        .page_content  → the text on that page
        .metadata      → {"source": "paper.pdf", "page": 3}

    Pages with no extractable text (e.g. scanned/image-only pages) are
    dropped so empty chunks never reach the vectorstore.
    """
    logger.info(f"Loading PDF: {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    usable = [doc for doc in documents if doc.page_content.strip()]
    skipped = len(documents) - len(usable)
    if skipped:
        logger.warning(
            f"Skipped {skipped} of {len(documents)} pages without text in "
            f"{pdf_path.name} (likely scanned/image-only content)"
        )
    if not usable:
        logger.error(f"No extractable text in {pdf_path.name} — not ingesting it")
        return []

    logger.info(f"Loaded {len(usable)} pages from {pdf_path.name}")
    return usable


def chunk_documents(documents: list) -> list:
    """
    Splits Document pages into smaller overlapping chunks.
    Returns a larger list of smaller Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,       # 1000 from config
        chunk_overlap=settings.chunk_overlap, # 200 from config
        length_function=len,
        # The separators to try, in order of preference:
        # 1. "\n\n" — paragraph break (best split point)
        # 2. "\n"   — line break
        # 3. " "    — word boundary
        # 4. ""     — character (last resort, never ideal)
        separators=["\n\n", "\n", " ", ""],
    )
    # .split_documents() takes a list of Documents and returns
    # a longer list of smaller Documents
    # Crucially: metadata is PRESERVED and COPIED to every chunk
    # So chunk 7 from page 3 still knows it came from page 3
    chunks = splitter.split_documents(documents)

    # Tag each chunk with its position within its page so later quality
    # upgrades (metadata-aware retrieval, finer citations) have data to use.
    per_page: dict = {}
    for chunk in chunks:
        key = (chunk.metadata.get("source"), chunk.metadata.get("page"))
        per_page[key] = per_page.get(key, -1) + 1
        chunk.metadata["chunk_index"] = per_page[key]

    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


def compute_chunk_id(doc) -> str:
    """
    Deterministic ID for a chunk, derived from its content and position.

    Re-ingesting the same PDF produces the same IDs, so storage can upsert
    (delete old IDs, add new) instead of appending duplicate chunks.
    """
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "?")
    content = " ".join(doc.page_content.split()).encode("utf-8")
    digest = hashlib.sha256(f"{source}|{page}|".encode("utf-8") + content).hexdigest()
    return f"{Path(source).name}:{page}:{digest[:16]}"


def embed_and_store(chunks: list, collection_name: str = "medmind") -> Chroma:
    """
    Takes chunks, embeds them, and stores them in ChromaDB.
    Returns the Chroma vectorstore object for querying.
    collection_name lets you have multiple separate knowledge bases
    in the same ChromaDB — e.g. "cardiology", "oncology", "neurology"

    Chunks are stored with deterministic IDs (see compute_chunk_id) and
    pre-existing IDs are removed first, so ingesting the same PDF twice
    updates its chunks instead of duplicating them.
    """
    logger.info(f"Embedding {len(chunks)} chunks into collection '{collection_name}'")
    embeddings = get_embeddings()

    ids = [compute_chunk_id(chunk) for chunk in chunks]
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        # str() because Chroma expects a string, not a Path object
        persist_directory=str(VECTORSTORE_DIR),
    )

    # Upsert semantics: chroma's add() errors on duplicate IDs, so drop any
    # IDs already stored before adding. Deleting IDs that don't exist is a no-op.
    if ids:
        vectorstore.delete(ids=ids)
    vectorstore.add_documents(documents=chunks, ids=ids)

    logger.info(f"Stored {len(chunks)} chunks in ChromaDB at {VECTORSTORE_DIR}")
    return vectorstore


# ─── Master function: do everything ──────────────────────────────────────────
def ingest_pdf(pdf_path: Path, collection_name: str = "medmind") -> Chroma:
    """
    Full pipeline: PDF path → chunked → embedded → stored in ChromaDB.
    This is the only function the rest of the project needs to call.
    Usage:
        from src.ingestion import ingest_pdf
        vectorstore = ingest_pdf(Path("data/papers/study.pdf"))
    """
    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents)
    vectorstore = embed_and_store(chunks, collection_name)
    return vectorstore


# ─── Load existing vectorstore (without re-ingesting) ─────────────────────────
@lru_cache(maxsize=16)
def load_vectorstore(collection_name: str = "medmind") -> Chroma:
    """
    Loads an already-existing ChromaDB collection from disk.
    Use this after the first ingest — no need to re-embed every time you run.

    The result is cached per collection so retrieval reuses the same
    vectorstore instead of reopening it on every question.
    The cache is cleared in delete_collection() so a deleted collection
    is never served stale from here.

    The flow in your app will be:
        First run  → ingest_pdf()        (slow, embeds everything)
        Every run after → load_vectorstore()  (fast, just loads from disk)
    """
    embeddings = get_embeddings()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
# ─── Collection management ────────────────────────────────────────────────────
def list_collections() -> list[str]:
    """
    Returns a list of all existing collections in ChromaDB.
    The UI uses this to show the user their existing knowledge bases.
    """

    # Connect directly to the ChromaDB client on disk
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    # Each collection is a separate knowledge base
    # .name gives us just the string name
    return [col.name for col in client.list_collections()]


def delete_collection(collection_name: str) -> None:
    """
    Deletes a collection and all its embeddings from ChromaDB.
    The UI uses this to let users reset and re-upload.
    """

    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    client.delete_collection(collection_name)

    # The cached vectorstore points at the now-deleted collection.
    load_vectorstore.cache_clear()
    logger.info(f"Deleted collection: {collection_name}")


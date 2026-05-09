# src/ingestion.py
import sys
import logging
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






def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns the local HuggingFace embedding model.
    First call downloads ~80MB model. Subsequent calls use cache.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_pdf(pdf_path: Path) -> list:
    """
    Loads a PDF and returns a list of Document objects, one per page.

    Each Document has:
        .page_content  → the text on that page
        .metadata      → {"source": "paper.pdf", "page": 3}
    """
    logger.info(f"Loading PDF: {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} pages from {pdf_path.name}")
    return documents


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
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


def embed_and_store(chunks: list, collection_name: str = "medmind") -> Chroma:
    """
    Takes chunks, embeds them, and stores them in ChromaDB.
    Returns the Chroma vectorstore object for querying.
    collection_name lets you have multiple separate knowledge bases
    in the same ChromaDB — e.g. "cardiology", "oncology", "neurology"
    """
    logger.info(f"Embedding {len(chunks)} chunks into collection '{collection_name}'")
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,

        # str() because Chroma expects a string, not a Path object
        persist_directory=str(VECTORSTORE_DIR),
    )

    logger.info(f"Stored in ChromaDB at {VECTORSTORE_DIR}")
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
def load_vectorstore(collection_name: str = "medmind") -> Chroma:
    """
    Loads an already-existing ChromaDB collection from disk.
    Use this after the first ingest — no need to re-embed every time you run.

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
    logger.info(f"Deleted collection: {collection_name}")


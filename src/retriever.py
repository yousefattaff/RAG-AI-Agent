# src/retriever.py
from pathlib import Path
from config import settings
# RunnablePassthrough lets a value pass through a chain unchanged
# We'll use it to forward the original question alongside the retrieved context
# Without it, the question would be "consumed" by the retriever and lost
from langchain_core.runnables import RunnablePassthrough

# AIMessage → plain string
from langchain_core.output_parsers import StrOutputParser

# ChatPromptTemplate builds our prompt with variables
from langchain_core.prompts import ChatPromptTemplate

# Document is the type that ChromaDB returns when you search
# Each Document has .page_content (the chunk text) and .metadata (source, page)
from langchain_core.documents import Document

# Our LLM factory and vectorstore loader
from src.llm import get_groq_llm
from src.ingestion import load_vectorstore

import logging
logger = logging.getLogger(__name__)


# ─── The RAG prompt ───────────────────────────────────────────────────────────
# This is the most important prompt in the whole project
# Notice it has TWO variables: {context} and {question}
# {context} = the retrieved chunks from ChromaDB (the documents)
# {question} = what the user actually asked
#
# The instruction "ONLY use the context below" is critical
# Without it, the LLM uses its training data and may hallucinate
# With it, the LLM is forced to stay grounded in YOUR documents

def build_rag_prompt(domain: str = "medical research") -> ChatPromptTemplate:
    """
    Builds a RAG prompt tailored to the user's chosen domain.

    domain examples:
        "medical research"
        "contract law"
        "financial analysis"
        "machine learning research"
        "whatever the user types"
    """
    return ChatPromptTemplate.from_messages([
("system", f"""You are an expert AI assistant specializing in {domain}.

Answer ONLY using the context below. 
If the context does not contain the answer, respond with:
"I couldn't find this in the provided documents."

Do NOT use any knowledge outside the provided context.
Always mention the source document and page number.

Context:
{{context}}
"""),
        ("human", "{question}"),
    ])


# ─── Helper: format documents into a readable context string ──────────────────
def format_docs(docs: list[Document]) -> str:
    """
    Takes a list of Document objects from ChromaDB and formats them
    into a single string that gets injected into the {context} variable.

    We include the source and page so the LLM can cite them in its answer.

    Example output:
        [Source: study.pdf | Page: 3]
        The maximum safe dosage is 500mg per day...

        [Source: study.pdf | Page: 7]
        Adverse effects were observed in 12% of patients...
    """
    formatted = []

    for doc in docs:
        # Pull metadata — .get() with a default so we never crash on missing keys
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")

        # Clean up the source path — we only want the filename, not the full path
        # Path(source).name turns "/home/user/data/papers/study.pdf" → "study.pdf"
        filename = Path(source).name

        formatted.append(
            f"[Source: {filename} | Page: {page}]\n{doc.page_content}"
        )

    # Join all chunks with a clear separator so the LLM can distinguish them
    return "\n\n---\n\n".join(formatted)


# ─── Build the retriever ──────────────────────────────────────────────────────
def get_retriever(collection_name: str = "medmind"):
    """
    Builds the retriever for a collection. Shared by the document-only RAG
    chain and the agent so retrieval settings stay consistent in both paths.

    Homogeneous results come from Maximal Marginal Relevance: MMR fetches a
    wider candidate pool (fetch_k) and then re-ranks it, balancing relevance
    against diversity. That prevents three near-identical chunks from
    occupying every k slot when the collection contains a lot of redundancy.
    """
    vectorstore = load_vectorstore(collection_name)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retrieval_top_k,
            "fetch_k": max(settings.retrieval_top_k * 4, 20),
        },
    )


# ─── Build the RAG chain ──────────────────────────────────────────────────────
def build_rag_chain(collection_name: str = "medmind", domain: str = "medical research"):

    retriever = get_retriever(collection_name)

    # Now uses the dynamic prompt instead of the hardcoded one
    prompt = build_rag_prompt(domain)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_groq_llm()
        | StrOutputParser()
    )

    return rag_chain


def _answer_chain(domain: str):
    """
    Prompt + LLM + parser only. Used by ask_with_sources when the retrieved
    documents are already in hand, so retrieval happens exactly once.
    """
    return build_rag_prompt(domain) | get_groq_llm() | StrOutputParser()


# ─── Retriever with sources ───────────────────────────────────────────────────
def ask_with_sources(
    question: str,
    collection_name: str = "medmind",
    domain: str = "medical research"
) -> dict:

    retriever = get_retriever(collection_name)
    source_docs = retriever.invoke(question)

    # Nothing usable retrieved → answer without spending an LLM call.
    # The canned response matches the fallback the prompt requires.
    if not source_docs:
        return {"answer": "I couldn't find this in the provided documents.", "sources": []}

    answer = _answer_chain(domain).invoke(
        {"context": format_docs(source_docs), "question": question}
    )

    sources = [
        {
            "filename": Path(doc.metadata.get("source", "unknown")).name,
            "page": doc.metadata.get("page", "?"),
            "excerpt": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
        }
        for doc in source_docs
    ]

    return {"answer": answer, "sources": sources}

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


# ─── Build the RAG chain ──────────────────────────────────────────────────────
def build_rag_chain(collection_name: str = "medmind", domain: str = "medical research"):
    
    vectorstore = load_vectorstore(collection_name)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k},
    )

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


# ─── Retriever with sources ───────────────────────────────────────────────────
def ask_with_sources(
    question: str,
    collection_name: str = "medmind",
    domain: str = "medical research"
) -> dict:
    
    vectorstore = load_vectorstore(collection_name)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k},
    )

    source_docs = retriever.invoke(question)
    chain = build_rag_chain(collection_name, domain)  # pass domain through
    answer = chain.invoke(question)

    sources = []
    for doc in source_docs:
        from pathlib import Path
        sources.append({
            "filename": Path(doc.metadata.get("source", "unknown")).name,
            "page": doc.metadata.get("page", "?"),
            "excerpt": doc.page_content[:200] + "...",
        })

    return {"answer": answer, "sources": sources}
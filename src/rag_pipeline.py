"""Advanced RAG retrieval built from the techniques in rag_techniques/.

Pipeline per question
---------------------
1. Query transformations + HyDE (one structured LLM call)
2. Fusion retrieval: vector MMR + BM25 (keyword)
3. HyDE / rewritten / step-back / sub-query candidate merge
4. Cross-encoder reranking
5. Context-window enrichment around each hit
6. CRAG-style relevance grading
7. Extractive compression of each chunk
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings
from src.ingestion import load_vectorstore
from src.llm import get_groq_llm
from src.rag_utils import (
    compress_chunk,
    expand_with_neighbors,
    reciprocal_rank_fusion,
    unique_keep_order,
)

logger = logging.getLogger(__name__)


class QueryPlan(BaseModel):
    rewritten_query: str = Field(description="More specific retrieval query")
    step_back_query: str = Field(description="Broader background query")
    sub_queries: list[str] = Field(
        default_factory=list,
        description="2-3 simpler sub-questions",
    )
    hypothetical_document: str = Field(
        description="Short hypothetical answer passage for HyDE retrieval",
    )


class DocGrades(BaseModel):
    scores: list[float] = Field(
        description="Relevance of each numbered chunk to the question, 0 to 1",
    )


def _cheap_llm():
    return get_groq_llm(temperature=0.0, model=settings.agent_groq_model)


def plan_query(question: str, history: list[dict] | None = None) -> QueryPlan:
    """Rewrite, step-back, decompose, and invent a HyDE passage in one call."""
    history_block = ""
    if history:
        turns = []
        for msg in history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            turns.append(f"{role}: {content}")
        history_block = "Recent conversation:\n" + "\n".join(turns) + "\n\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You improve retrieval queries for a document RAG system. "
         "Use conversation history if the latest question is a follow-up. "
         "Keep the hypothetical document short (under 800 characters) and "
         "in the style of a textbook passage, not a chatbot answer."),
        ("human",
         history_block + "Latest question: {question}"),
    ])
    try:
        chain = prompt | _cheap_llm().with_structured_output(QueryPlan)
        plan = chain.invoke({"question": question})
        if not plan.rewritten_query:
            plan.rewritten_query = question
        plan.sub_queries = [q for q in plan.sub_queries if q.strip()][:3]
        return plan
    except Exception as e:
        logger.warning(f"Query planning failed, using the original question: {e}")
        return QueryPlan(
            rewritten_query=question,
            step_back_query=question,
            sub_queries=[],
            hypothetical_document=question,
        )


@lru_cache(maxsize=16)
def _collection_corpus(collection_name: str) -> tuple[Document, ...]:
    store = load_vectorstore(collection_name)
    data = store.get(include=["metadatas", "documents"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or [{}] * len(documents)
    docs = [
        Document(page_content=text or "", metadata=meta or {})
        for text, meta in zip(documents, metadatas)
        if text
    ]
    return tuple(docs)


def clear_retrieval_caches() -> None:
    _collection_corpus.cache_clear()


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _vector_candidates(collection_name: str, query: str, k: int) -> list[Document]:
    store = load_vectorstore(collection_name)
    fetch_k = max(k * 4, settings.retrieval_fetch_k)
    try:
        return store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k,
        )
    except Exception as e:
        logger.warning(f"MMR search failed, falling back to similarity: {e}")
        return store.similarity_search(query, k=k)


def _bm25_candidates(collection_name: str, query: str, k: int) -> list[Document]:
    corpus = list(_collection_corpus(collection_name))
    if not corpus:
        return []
    retriever = BM25Retriever.from_documents(corpus)
    retriever.k = k
    return retriever.invoke(query)


def _merge_query_candidates(
    collection_name: str,
    question: str,
    plan: QueryPlan,
    k: int,
) -> list[Document]:
    queries = [question, plan.rewritten_query, plan.step_back_query]
    queries.extend(plan.sub_queries)
    ranked_lists: list[list[Document]] = []

    for q in unique_keep_order(q for q in queries if q and q.strip()):
        ranked_lists.append(_vector_candidates(collection_name, q, k))
        ranked_lists.append(_bm25_candidates(collection_name, q, k))

    if plan.hypothetical_document and plan.hypothetical_document.strip() != question:
        store = load_vectorstore(collection_name)
        ranked_lists.append(
            store.similarity_search(plan.hypothetical_document, k=k)
        )

    return reciprocal_rank_fusion(ranked_lists)


def rerank_cross_encoder(
    query: str,
    docs: list[Document],
    top_n: int,
) -> list[Document]:
    if not docs:
        return []
    if len(docs) == 1:
        return docs
    try:
        model = _cross_encoder()
        pairs = [[query, doc.page_content[:4000]] for doc in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda item: float(item[1]), reverse=True)
        for doc, score in ranked:
            doc.metadata["rerank_score"] = float(score)
        return [doc for doc, _ in ranked[:top_n]]
    except Exception as e:
        logger.warning(f"Cross-encoder rerank failed: {e}")
        return docs[:top_n]


def grade_documents(question: str, docs: list[Document]) -> list[Document]:
    """Drop chunks the grader marks as irrelevant (CRAG / reliable RAG)."""
    if not docs:
        return []
    numbered = "\n\n".join(
        f"[{i}] {doc.page_content[:1200]}" for i, doc in enumerate(docs)
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Score how relevant each numbered chunk is to the question. "
         "Use 0 for unrelated, 1 for directly useful. "
         "Return one score per chunk, in the same order."),
        ("human", "Question: {question}\n\nChunks:\n{chunks}"),
    ])
    try:
        chain = prompt | _cheap_llm().with_structured_output(DocGrades)
        grades = chain.invoke({"question": question, "chunks": numbered})
        scores = list(grades.scores)
        kept = []
        for i, doc in enumerate(docs):
            score = scores[i] if i < len(scores) else 0.0
            doc.metadata["relevance_grade"] = score
            if score >= settings.relevance_threshold:
                kept.append(doc)
        return kept if kept else docs[:1]
    except Exception as e:
        logger.warning(f"Document grading failed, keeping reranked docs: {e}")
        return docs


def retrieve_advanced(
    question: str,
    collection_name: str,
    history: list[dict] | None = None,
) -> list[Document]:
    k = settings.retrieval_candidate_k
    top_k = settings.retrieval_top_k

    plan = plan_query(question, history)
    logger.info(
        "Query plan rewritten=%r step_back=%r sub=%s",
        plan.rewritten_query,
        plan.step_back_query,
        plan.sub_queries,
    )

    fused = _merge_query_candidates(collection_name, question, plan, k)
    if not fused:
        return []

    reranked = rerank_cross_encoder(plan.rewritten_query or question, fused, top_n=top_k)
    corpus = list(_collection_corpus(collection_name))
    enriched = expand_with_neighbors(
        reranked, corpus, num_neighbors=settings.context_neighbors,
    )
    graded = grade_documents(question, enriched)

    compressed: list[Document] = []
    for doc in graded:
        copy = doc.model_copy()
        copy.page_content = compress_chunk(question, doc.page_content)
        compressed.append(copy)
    return compressed

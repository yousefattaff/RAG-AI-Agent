# src/agent.py

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from src.llm import get_groq_llm
from src.tools import STATIC_TOOLS
from src.ingestion import load_vectorstore
from src.retriever import format_docs
from config import settings
import logging

logger = logging.getLogger(__name__)


def retrieve_context(question: str, collection_name: str) -> str:
    """
    Retrieves relevant document chunks BEFORE the agent runs.
    This replaces the search_documents tool entirely.
    Documents are injected into the system prompt as context —
    the LLM reads them directly without needing to call a tool.
    """
    try:
        vectorstore = load_vectorstore(collection_name)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retrieval_top_k},
        )
        docs = retriever.invoke(question)

        if not docs:
            return "No relevant information found in the uploaded documents."

        return format_docs(docs)

    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        return "Could not retrieve documents."


def build_system_prompt(domain: str, context: str) -> str:
    return f"""You are an expert AI assistant specializing in {domain}.

The following context was retrieved from the uploaded documents.
Use it as your PRIMARY source of information:

---DOCUMENT CONTEXT---
{context}
---END CONTEXT---

STRICT RULES:
- Answer using the document context above first
- Only use tools (pubmed_search, web_search, medical_calculator)
  if the document context does not contain enough information
- If neither documents nor tools have the answer, say:
  "I don't have enough information in the provided documents to answer this."
- NEVER answer from your training knowledge
- NEVER discuss people, topics, or files not related to {domain}
- Always cite the source document and page number
"""


def ask_agent(
    question: str,
    collection_name: str = "medmind",
    domain: str = "general research",
) -> dict:
    try:
        # ── Step 1: Retrieve context BEFORE agent starts ───────────────────────
        # Documents are retrieved here and injected into the system prompt
        # The LLM reads them as part of its context — no tool call needed
        # This completely eliminates the <function=search_documents> error
        logger.info("Retrieving document context...")
        context = retrieve_context(question, collection_name)

        # ── Step 2: Only external tools (no search_documents) ─────────────────
        # STATIC_TOOLS = pubmed_search, web_search, medical_calculator
        # None of these cause the XML format error
        tools = STATIC_TOOLS
        tool_map = {tool.name: tool for tool in tools}

        # ── Step 3: Build LLM with tools bound ────────────────────────────────
        llm = get_groq_llm(temperature=0.0)
        llm_with_tools = llm.bind_tools(tools)

        # ── Step 4: Build messages with context already in system prompt ───────
        messages = [
            SystemMessage(content=build_system_prompt(domain, context)),
            HumanMessage(content=question),
        ]

        # ── Step 5: Agent loop ────────────────────────────────────────────────
        for iteration in range(settings.max_agent_iterations):
            logger.info(f"Agent iteration {iteration + 1}")

            response = llm_with_tools.invoke(messages)
            messages.append(response)

            # No tool calls → LLM is done
            if not response.tool_calls:
                logger.info("Agent finished")
                return {
                    "answer": response.content,
                    "error": None,
                }

            # Execute each tool
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                logger.info(f"Calling tool: {tool_name}")

                if tool_name in tool_map:
                    try:
                        tool_result = tool_map[tool_name].invoke(tool_args)
                    except Exception as e:
                        tool_result = f"Tool error: {str(e)}"
                else:
                    tool_result = f"Tool '{tool_name}' not found."

                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                ))

        return {
            "answer": "I wasn't able to find a complete answer. Please try rephrasing.",
            "error": None,
        }

    except Exception as e:
        logger.error(f"Agent failed: {e}")
        return {
            "answer": None,
            "error": str(e),
        }
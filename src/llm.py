# src/llm.py


from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from config import settings


def get_groq_llm(
    temperature: float = 0.2,
    model: str | None = None,
) -> ChatGroq:
    """
    Returns a ChatGroq instance ready to use.

    temperature controls randomness:
        0.0 = fully deterministic, same question always same answer
        1.0 = creative, varied answers
        0.1 = nearly deterministic — good for factual medical Q&A
              we don't want the LLM "getting creative" with diagnoses
    """
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=model or settings.groq_model,
        temperature=temperature,
    )


def get_ollama_llm(temperature: float = 0.1) -> ChatOllama:
    """
    Returns a ChatOllama instance pointing at your local Ollama server.
    Ollama must be running: `ollama serve` in a separate terminal.
    """
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=temperature,
    )
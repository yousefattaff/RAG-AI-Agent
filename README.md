# RAG-AI-Agent

A Streamlit application for building named “experts” from your PDFs: each expert is backed by a local ChromaDB collection, retrieval-augmented generation (RAG), and an optional LangChain tool-calling agent (PubMed, Tavily web search, and clinical calculators).

The goal is grounded answers tied to uploaded material, with citations to filename and page, plus external tools when **agent mode** is enabled.

---

## Table of contents

- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the application](#running-the-application)
- [Using the UI](#using-the-ui)
- [Project layout](#project-layout)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

---

## Capabilities

| Area | Behavior |
|------|----------|
| **Knowledge bases** | Upload PDFs per expert; each expert maps to its own ChromaDB collection under `vectorstore/`. |
| **RAG (default path)** | Similarity retrieval over chunked pages, Hugging Face embeddings locally, Groq Chat API for answering with document-only grounding. |
| **Agent mode** | Pre-retrieves document chunks into the system prompt, then Groq may call tools: PubMed (`Bio.Entrez`), web search (**Tavily**), `medical_calculator` (BMI, eGFR, BSA). |
| **Speech** | Optional neural TTS via **Microsoft Edge TTS** (`edge-tts`) when “Voice responses” is enabled (requires outbound network access). |

Non-goals today: scanned PDFs are not OCR’d in the ingestion pipeline (`PyPDFLoader` only); optional OCR-related dependencies do not enable that path yet.

---

## Architecture

**Ingestion**

1. `PyPDFLoader` → one LangChain document per PDF page (with `source`, `page` metadata).
2. `RecursiveCharacterTextSplitter` using `chunk_size` / `chunk_overlap` from settings.
3. `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2` by default; 384-dimensional normalized vectors).
4. `Chroma` with `persist_directory` set to `./vectorstore`.

**Document-only answering (`ask_with_sources`)**

- Retrieves top‑`k` chunks, builds context with `[Source: filename | Page: n]`, invokes Groq with a strict “context only” system prompt.
- UI shows a collapsible source list driven by retrieval metadata.

**Agent answering (`ask_agent`)**

- Retrieves top‑`k` chunks first and injects them into the system prompt (no separate `search_documents` tool call in this flow).
- Binds PubMed, Tavily web search, and the calculator tools; runs a bounded iteration loop (`max_agent_iterations`).
- Structured “Sources” UI expander is not populated on the agent path; the model is instructed to cite filename and page in the answer text when using document context.

---

## Requirements

- **Python** 3.12 or newer (see `pyproject.toml`).
- **Groq** API key for the chat model.
- **Tavily** API key — required at startup because `Settings` validates it (`config.py`); only used when the agent invokes `web_search`.

Optional: NVIDIA GPU improves embedding speed (`torch.cuda` if available).

---

## Installation

Clone the repository and install dependencies with **uv** (recommended):

```bash
git clone https://github.com/yousefattaff/RAG-AI-Agent.git
cd RAG-AI-Agent
uv sync
```

This installs the packages declared in `pyproject.toml`. A legacy `requirements.txt` exists but may not match that full set.

### Windows activation (if using a manual venv)

```powershell
uv venv
.venv\Scripts\Activate.ps1
uv sync
```

### macOS / Linux

```bash
uv venv
source .venv/bin/activate
uv sync
```

---

## Configuration

Environment variables are loaded from a `.env` file in the project root via `config.py` (Pydantic Settings).

### Required

| Variable | Purpose |
|---------|---------|
| `GROQ_API_KEY` | Groq Chat API authentication. |
| `TAVILY_API_KEY` | Validated at import; consumed by Tavily when the agent calls `web_search`. |

### Optional

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_GPT_GROQ` | `llama-3.3-70b-versatile` | Groq model for the RAG chain and for the agent (both call `get_groq_llm` without overriding `model`). |
| `MODEL_AGENT_GROQ` | `llama-3.1-8b-instant` | Declared in `config.py`; not passed through in `src/agent.py` today. To use a separate agent model, wire `model=settings.agent_groq_model` in `get_groq_llm` calls. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Available via `get_ollama_llm` in `src/llm.py`; the Streamlit app uses Groq only. |
| `OLLAMA_MODEL` | `llama3.2` | Same note as above. |

Embedding model name is set in code (`Settings.embedding_model`), default **`all-MiniLM-L6-v2`**. Extend `Settings` if you need it read from `.env`.

Other tunables live on `settings` in `config.py`:

| Setting | Default | Role |
|---------|---------|------|
| `chunk_size` | `1000` | Text splitter |
| `chunk_overlap` | `200` | Text splitter |
| `retrieval_top_k` | `3` | Chunks retrieved for RAG and agent context injection |
| `max_agent_iterations` | `5` | Tool-call loop ceiling |

PubMed usage follows NCBI Entrez etiquette; **`Entrez.email` is set in `src/tools.py`** — replace with your own contact address for production or shared deployments.

Example `.env` skeleton:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
MODEL_GPT_GROQ=llama-3.3-70b-versatile
```

---

## Running the application

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Using the UI

1. **Create expert** — Sidebar: expert name (becomes collection name after normalization), domain description, PDF uploads, then **Build Expert**.
2. **Load expert** — Select an existing collection or delete one.
3. **Agentic mode** — On: PubMed / Tavily / calculator tools after document context is injected. Off: faster document-only RAG path with explicit source list in the UI.
4. **Voice responses** — When enabled, responses are synthesized with Edge TTS and autoplay helper HTML (requires network).

---

## Project layout

```
RAG-AI-Agent/
├── app.py              # Streamlit UI and session state
├── config.py           # Paths, pydantic-settings, defaults
├── pyproject.toml      # Dependencies (uv / PEP 621)
├── requirements.txt   # Legacy; prefer pyproject.toml + uv sync
├── data/papers/        # Uploaded PDF storage (ensure write access)
├── vectorstore/       # Persistent Chroma data
└── src/
    ├── ingestion.py    # Load PDFs, chunk, embed, persist
    ├── retriever.py    # RAG chain + ask_with_sources
    ├── agent.py        # Tool-calling Groq agent with injected context
    ├── tools.py        # PubMed, Tavily, calculator; document-search factory (unused by current agent wiring)
    ├── llm.py          # ChatGroq / ChatOllama factories
    └── tts.py          # Edge TTS helpers
```

---

## Limitations

- Groq free tier throughput is finite; bursts may trigger API errors depending on account limits.
- Tavily and Edge TTS require internet access; embeddings and Chroma are local.
- PDFs that are image-only scans need an OCR ingestion path (see roadmap); current loader reads embedded text only.

---

## Roadmap

- OCR or vision pipeline for scanned PDFs
- Streaming token responses in the UI
And i am open to any new ideas :)

---

## Author

**Yousef Alsayed**  

Update the placeholders with your profiles when publishing:

- GitHub: `https://github.com/yourusername`
- LinkedIn: `https://linkedin.com/in/yourusername`


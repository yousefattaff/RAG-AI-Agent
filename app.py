# app.py
# Run with: streamlit run app.py  OR  uv run streamlit run app.py

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import logging
import time

from src.ingestion import ingest_pdf, list_collections, delete_collection
from src.retriever import ask_with_sources
from src.agent import ask_agent
from src.tts import text_to_speech_bytes, autoplay_audio_html, VOICES
from config import PAPERS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG-AI-Agent",
    page_icon="🌵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session state ────────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "collection_name": None,
    "domain": None,
    "agent_ready": False,
    "tts_enabled": False,
    "use_agent": True,
    "selected_voice": VOICES["en-female"],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌵 RAG-AI-Agent")
    st.caption("Document Intelligence Agent")
    st.divider()

    mode = st.radio(
        "Mode",
        ["Create new expert", "Load existing expert"],
    )

    st.divider()

    # ── Create new expert ─────────────────────────────────────────────────────
    if mode == "Create new expert":
        st.subheader("Configure your expert")

        expert_name = st.text_input(
            "Expert name",
            placeholder="",
        )

        domain = st.text_input(
            "Domain / specialization",
            placeholder="",
        )

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf"],
            accept_multiple_files=True,
        )

        build_disabled = not (expert_name and domain and uploaded_files)

        if st.button("Build expert", type="primary", disabled=build_disabled):

            collection_name = expert_name.lower().replace(" ", "_")

            with st.spinner(f"Building {expert_name}..."):
                progress = st.progress(0)

                for i, uploaded_file in enumerate(uploaded_files):
                    st.write(f"Processing: {uploaded_file.name}")

                    temp_path = PAPERS_DIR / uploaded_file.name
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        ingest_pdf(temp_path, collection_name)
                    except Exception as e:
                        logger.error(f"Ingestion failed for {uploaded_file.name}: {e}")
                        st.warning(f"Skipped {uploaded_file.name} — could not process.")

                    progress.progress((i + 1) / len(uploaded_files))

                    if i < len(uploaded_files) - 1:
                        time.sleep(1)

                st.session_state.collection_name = collection_name
                st.session_state.domain = domain
                st.session_state.agent_ready = True
                st.session_state.messages = []

            st.success(f"{expert_name} is ready!")
            st.rerun()

    # ── Load existing expert ──────────────────────────────────────────────────
    else:
        st.subheader("Your experts")

        existing = list_collections()

        if not existing:
            st.info("No experts yet. Create one first.")
        else:
            selected = st.selectbox(
                "Select an expert",
                existing,
                format_func=lambda x: x.replace("_", " ").title(),
            )

            domain = st.text_input(
                "Domain (re-enter for this session)",
                placeholder="e.g. diabetes research",
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Load", type="primary", disabled=not domain):
                    st.session_state.collection_name = selected
                    st.session_state.domain = domain
                    st.session_state.agent_ready = True
                    st.session_state.messages = []
                    st.rerun()

            with col2:
                if st.button("Delete", type="secondary"):
                    delete_collection(selected)
                    st.warning(f"Deleted {selected}")
                    st.rerun()

    # ── Active expert settings ─────────────────────────────────────────────────
    if st.session_state.agent_ready:
        st.divider()
        st.success(
            f"Active: {st.session_state.collection_name.replace('_', ' ').title()}"
        )
        st.caption(f"Domain: {st.session_state.domain}")

        st.session_state.tts_enabled = st.toggle(
            "Voice responses",
            value=st.session_state.tts_enabled,
            help="Read answers aloud automatically",
        )

        if st.session_state.tts_enabled:
            selected_voice_key = st.selectbox(
                "Voice",
                options=list(VOICES.keys()),
                format_func=lambda x: x.replace("-", " ").title(),
            )
            st.session_state.selected_voice = VOICES[selected_voice_key]

        st.session_state.use_agent = st.toggle(
            "Agentic mode",
            value=st.session_state.use_agent,
            help="ON: uses tools (PubMed, web, calculator). OFF: documents only",
        )

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()


# ─── Main area ────────────────────────────────────────────────────────────────
if not st.session_state.agent_ready:

    st.title("Welcome to 🌵 RAG-AI-Agent")
    st.subheader("Your personal document intelligence agent")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📄 **Upload any PDFs**\n\nBuilds a searchable knowledge base from your documents")
    with col2:
        st.info("🔍 **Ask anything**\n\nAgent searches documents, PubMed, and the web")
    with col3:
        st.info("🔊 **Hear the answers**\n\nNeural text-to-speech reads responses aloud")

    st.divider()
    st.markdown("**Get started:** Use the sidebar to create your first expert →")

else:

    # ── Chat header ───────────────────────────────────────────────────────────
    expert_display = st.session_state.collection_name.replace("_", " ").title()
    st.title(f"🧠 {expert_display}")
    st.caption(f"Specialized in: {st.session_state.domain}")

    # ── Chat history 
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Sources", expanded=False):
                    for source in message["sources"]:
                        st.caption(
                            f"📄 **{source['filename']}** — Page {source['page']}"
                        )
                        st.caption(f"> {source['excerpt']}")

    #Text input only
    prompt = st.chat_input("Ask your expert anything...")

    # ── Generate response ─────────────────────────────────────────────────────
    if prompt:

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                sources = []
                answer = None

                if st.session_state.use_agent:
                    result = ask_agent(
                        question=prompt,
                        collection_name=st.session_state.collection_name,
                        domain=st.session_state.domain,
                    )

                    if result.get("error") or not result.get("answer"):
                        actual_error = result.get('error', 'No answer returned')
                        logger.error(f"Agent error: {actual_error}")
                        # Show real error during development so we can debug
                        answer = f"Something went wrong: {actual_error}"
                    else:
                        answer = result["answer"]

                else:
                    try:
                        result = ask_with_sources(
                            question=prompt,
                            collection_name=st.session_state.collection_name,
                            domain=st.session_state.domain,
                        )
                        answer = result["answer"]
                        sources = result["sources"]
                    except Exception as e:
                        logger.error(f"RAG error: {e}")
                        answer = (
                            "I couldn't retrieve an answer right now. "
                            "Please try again in a moment."
                        )

            # Display answer
            st.markdown(answer)

            # Sources
            if sources:
                with st.expander("Sources", expanded=False):
                    for s in sources:
                        st.caption(f"📄 **{s['filename']}** — Page {s['page']}")
                        st.caption(f"> {s['excerpt']}")

            # TTS autoplay
            if st.session_state.tts_enabled and answer:
                with st.spinner("Generating voice..."):
                    try:
                        audio_response = text_to_speech_bytes(
                            answer,
                            voice=st.session_state.selected_voice,
                        )
                        if audio_response:
                            components.html(
                                autoplay_audio_html(audio_response),
                                height=0,
                            )
                    except Exception as e:
                        logger.error(f"TTS failed: {e}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
# src/tts.py

import asyncio
import edge_tts
from pathlib import Path
from io import BytesIO
import re
import logging

logger = logging.getLogger(__name__)

# Microsoft neural voices — pick one
# Full list: run `edge-tts --list-voices` in terminal
VOICES = {
    "en-female": "en-US-JennyNeural",     # natural American female
    "en-male":   "en-US-GuyNeural",       # natural American male
    "en-uk":     "en-GB-SoniaNeural",     # British female
    "ar-female": "ar-EG-SalmaNeural",     # Arabic female
    "ar-male":   "ar-EG-ShakirNeural",    # Arabic male
}

# Default voice
DEFAULT_VOICE = VOICES["en-female"]


def _clean_text_for_speech(text: str) -> str:
    """Remove markdown that sounds bad when read aloud."""
    # Remove bold/italic markers
    text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove source citations [Source: x | Page: 3]
    text = re.sub(r'\[Source:[^\]]+\]', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove horizontal rules
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    # Collapse extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


async def _generate_audio_async(text: str, voice: str) -> bytes:
    """
    Async function that generates audio bytes from text.
    edge-tts is async natively — this is why we use asyncio.
    """
    communicate = edge_tts.Communicate(text, voice)

    audio_chunks = []

    # .stream() yields chunks as they're generated
    # chunk["type"] == "audio" means it's actual audio data
    # chunk["type"] == "WordBoundary" is timing metadata — we skip it
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])

    # Join all chunks into one bytes object
    return b"".join(audio_chunks)


def text_to_speech_bytes(
    text: str,
    voice: str = DEFAULT_VOICE,
) -> bytes:
    """
    Converts text to speech and returns raw MP3 bytes.
    Streamlit uses this to play audio inline with st.audio().

    Returns:
        MP3 audio as bytes — pass directly to st.audio()
    """
    clean = _clean_text_for_speech(text)

    if not clean:
        return b""

    # asyncio.run() runs the async function synchronously
    # We need this because Streamlit is not async
    audio_bytes = asyncio.run(_generate_audio_async(clean, voice))

    logger.info(f"Generated {len(audio_bytes)} bytes of audio")
    return audio_bytes


def text_to_speech_file(
    text: str,
    save_path: Path,
    voice: str = DEFAULT_VOICE,
) -> Path:
    """Saves speech to an MP3 file and returns the path."""
    audio_bytes = text_to_speech_bytes(text, voice)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(audio_bytes)
    return save_path

def autoplay_audio_html(audio_bytes: bytes) -> str:
    """
    Returns an HTML string with an invisible autoplaying audio element.
    Uses base64 encoding so no file is needed — the audio lives in the HTML.

    Why base64? Browsers block autoplay on external file URLs for security.
    Embedding the audio as a data URL bypasses this restriction.
    """
    import base64

    # base64 encodes binary bytes as a safe ASCII string
    # decode("utf-8") converts bytes → string for the HTML
    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

    # The audio element is hidden (display:none) — no player UI shown
    # autoplay triggers immediately when the HTML loads
    # type="audio/mpeg" tells the browser this is MP3
    return f"""
        <audio autoplay style="display:none">
            <source src="data:audio/mpeg;base64,{b64_audio}" type="audio/mpeg">
        </audio>
    """
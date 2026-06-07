"""
Purpose:
Handles Speech-to-Text (Whisper) and Text-to-Speech (Edge TTS).

Why:
Enables voice-to-voice interaction alongside text input.

Functions:
  transcribe_audio(path) -> str        : audio file -> text
  text_to_speech(text)   -> bytes      : text -> MP3 bytes
"""

import whisper
import edge_tts
import asyncio
import tempfile
import os

WHISPER_MODEL = "base"   # tiny / base / small / medium
TTS_VOICE     = "en-US-AriaNeural"   # Edge TTS voice

# Load Whisper model once at module level (cached)
_whisper_model = None

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    """
    Convert recorded audio file to text using Whisper.

    Args:
        audio_path: path to .wav / .mp3 / .webm audio file

    Returns:
        Transcribed text string
    """
    model = _get_whisper()
    result = model.transcribe(audio_path, fp16=False)
    return result["text"].strip()


async def _synthesize(text: str, out_path: str):
    """Internal async Edge TTS synthesizer."""
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    await communicate.save(out_path)


def text_to_speech(text: str) -> bytes:
    """
    Convert answer text to MP3 audio bytes using Edge TTS.

    Args:
        text: plain text answer (markdown stripped)

    Returns:
        MP3 audio as bytes (ready for st.audio)
    """
    # Strip markdown symbols for cleaner speech
    clean = (text
             .replace("**", "")
             .replace("##", "")
             .replace("#",  "")
             .replace("- ", "")
             .replace("*",  ""))

    # Write to a temp file, read back as bytes
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name

    try:
        asyncio.run(_synthesize(clean[:2000], tmp_path))   # cap length
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)
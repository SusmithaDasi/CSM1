import os
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

def text_to_speech(summary: str, voice_id: str = None) -> str:
    """
    Converts the blog summary into audio using ElevenLabs SDK v2.x.
    Allows choosing a custom voice.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ElevenLabs API key not found. Please set ELEVENLABS_API_KEY.")

    client = ElevenLabs(api_key=api_key)
    voices = client.voices.get_all().voices

    if not voice_id:
        voice_id = voices[0].voice_id if voices else "EXAVITQu4vr4xnSDxMaL"

    audio_stream = client.text_to_speech.convert(
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        text=summary,
        output_format="mp3_44100_128"
    )

    output_path = "podcast.mp3"
    with open(output_path, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)

    return output_path

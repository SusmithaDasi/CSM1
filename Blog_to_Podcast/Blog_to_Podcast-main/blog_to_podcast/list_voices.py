from elevenlabs.client import ElevenLabs
import os
from dotenv import load_dotenv

load_dotenv()
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

voices = client.voices.get_all().voices

print("\n🎤 Available Voices:")
for v in voices:
    print(f"- {v.voice_id}: {v.name}")

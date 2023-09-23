import os
import requests
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

def test_openai():
    print("\n🧠 Testing OpenAI API...")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("❌ OpenAI API key missing in .env")
        return
    try:
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'Hello from OpenAI API test!'"}]
        )
        print("✅ OpenAI API working:", response.choices[0].message.content.strip())
    except Exception as e:
        print("❌ OpenAI test failed:", e)

def test_firecrawl():
    print("\n🔥 Testing Firecrawl API...")
    key = os.getenv("FIRECRAWL_API_KEY")
    if not key:
        print("❌ Firecrawl API key missing in .env")
        return
    try:
        response = requests.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={"Authorization": f"Bearer {key}"},
            json={"url": "https://example.com"}
        )
        if response.status_code == 200:
            print("✅ Firecrawl API working!")
        else:
            print("❌ Firecrawl test failed:", response.text)
    except Exception as e:
        print("❌ Firecrawl test failed:", e)

def test_elevenlabs():
    print("\n🎙️ Testing ElevenLabs API (v2.x)...")
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        print("❌ ElevenLabs API key missing in .env")
        return
    try:
        client = ElevenLabs(api_key=key)

        # Get available voices
        voices = client.voices.get_all().voices
        if not voices:
            print("❌ No voices found in your ElevenLabs account.")
            return

        # Pick the first available voice
        test_voice = voices[0].voice_id
        print(f"🎤 Using voice: {voices[0].name}")

        # Generate a short test audio
        audio = client.text_to_speech.convert(
            voice_id=test_voice,
            model_id="eleven_multilingual_v2",
            text="This is a test from ElevenLabs API!",
            output_format="mp3_44100_128",
        )

        # Save audio output
        with open("test_audio.mp3", "wb") as f:
            for chunk in audio:
                f.write(chunk)

        print("✅ ElevenLabs API working! (test_audio.mp3 generated successfully)")
    except Exception as e:
        print("❌ ElevenLabs test failed:", e)

if __name__ == "__main__":
    print("🔍 Running API Key Tests...")
    test_openai()
    test_firecrawl()
    test_elevenlabs()
    print("\n✅ All tests complete!")

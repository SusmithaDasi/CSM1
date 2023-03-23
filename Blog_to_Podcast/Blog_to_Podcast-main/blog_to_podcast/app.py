import streamlit as st
from utils.scrape_blog import scrape_blog
from utils.summarize_blog import summarize_blog
from utils.generate_podcast import text_to_speech
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os

# ==========================
# 🔐 Secure Environment Setup
# ==========================
load_dotenv()

# Verify API keys
required_keys = ["OPENAI_API_KEY", "FIRECRAWL_API_KEY", "ELEVENLABS_API_KEY"]
missing = [k for k in required_keys if not os.getenv(k)]
if missing:
    st.error(f"❌ Missing API keys: {', '.join(missing)}. Please add them to your .env file.")
    st.stop()

# ==========================
# 🎧 App Configuration
# ==========================
st.set_page_config(page_title="Blog to Podcast Agent", page_icon="🎧", layout="wide")
st.title("🎙️ Blog to Podcast Agent")
st.write("Turn any blog into an engaging podcast using GPT-4, Firecrawl, and ElevenLabs.")

# ==========================
# 🎤 ElevenLabs Voice Selection
# ==========================
voice_id = None
try:
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    voices = client.voices.get_all().voices
    voice_names = {v.name: v.voice_id for v in voices}
    selected_voice = st.sidebar.selectbox("🎤 Select Voice", list(voice_names.keys()))
    voice_id = voice_names[selected_voice]
except Exception as e:
    st.sidebar.error(f"Could not fetch ElevenLabs voices: {e}")

# ==========================
# 🌐 Blog URL Input
# ==========================
url = st.text_input("🌍 Enter Blog URL:", placeholder="https://example.com/blog-post")

# ==========================
# 🚀 Main Logic
# ==========================
if st.button("🎧 Generate Podcast") and url:
    try:
        with st.spinner("🕸️ Scraping blog content..."):
            content = scrape_blog(url)
            st.text(f"Length of scraped text: {len(content)} characters")
            st.code(content[:500])  # show first 500 chars for debugging

            if not content or len(content.strip()) < 50:   # treat very short text as failure
                st.error("❌ Could not retrieve blog content.")
                st.write("Debug info ↓")
                st.code(content)        # <-- shows whatever the scraper got
                st.stop()


        with st.spinner("🧠 Summarizing blog content using GPT-4..."):
            summary = summarize_blog(content)
            st.subheader("📝 Blog Summary")
            st.write(summary)

        with st.spinner("🎤 Generating Podcast Audio..."):
            audio_path = text_to_speech(summary, voice_id=voice_id)
            st.success("✅ Podcast generated successfully!")
            st.audio(audio_path)

            with open(audio_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Podcast Audio",
                    f,
                    file_name="podcast.mp3",
                    mime="audio/mpeg",
                )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("Enter a blog URL above and click **🎧 Generate Podcast** to begin.")

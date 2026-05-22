# 🎧 Blog to Podcast Agent

## 📖 Description
**Blog to Podcast Agent** is an AI-powered Streamlit web application that transforms any public blog post into an engaging podcast.  
It uses **GPT-4** to summarize the content, **Firecrawl API** (or BeautifulSoup fallback) to scrape the blog text, and **ElevenLabs API** to generate natural-sounding podcast audio.  

Turn any blog into a voice that speaks — fast, intelligent, and human-like 🎙️

---

## 🌟 Features
- 🕸️ **Smart Blog Scraping:** Automatically extracts blog text using Firecrawl or fallback parser  
- 🧠 **AI Summarization (GPT-4):** Generates concise, listener-friendly versions of long blogs  
- 🎤 **Podcast Generation (ElevenLabs):** Produces realistic, human-like speech in seconds  
- 💾 **One-Click Download:** Download your podcast as an MP3 file instantly  
- 🔐 **Secure API Key Management:** Environment variables handled via `.env` or Streamlit Secrets  
- ⚙️ **Cross-Platform:** Works locally on any OS or on Streamlit Cloud deployment  

---

## 🧠 Tech Stack
| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **AI Model** | OpenAI GPT-4 |
| **Scraping** | Firecrawl API + BeautifulSoup |
| **Audio Generation** | ElevenLabs Voice API |
| **Language** | Python 3.10+ |
| **Deployment** | Streamlit Cloud |

---

## ⚙️ Installation (Local Setup)

```bash
git clone https://github.com/<your-username>/blog_to_podcast.git
cd blog_to_podcast
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Create a `.env` file:
```
OPENAI_API_KEY=sk-your-key
FIRECRAWL_API_KEY=fc-your-key
ELEVENLABS_API_KEY=sk-your-key
```

### Run the app:
```bash
streamlit run app.py
```
---

## 🧾 Project Structure
```
blog_to_podcast/
│
├── app.py
├── requirements.txt
├── .env
└── utils/
    ├── scrape_blog.py
    ├── summarize_blog.py
    └── generate_podcast.py
```
---

## 👩‍💻 Author

**Susmitha Dasi**  
📍 Hyderabad, Telangana, India.
📧 todssusmitha@gmail.com

---

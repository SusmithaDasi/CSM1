# 🧠 Browser MCP Agent

Control a browser using **natural language commands**!
Built with **Streamlit**, **Playwright**, and **OpenAI GPT-4** (async version), this agent supports:

* Multi-step tasks
* Automatic searches on YouTube, Google, Wikipedia
* Click, scroll, and screenshot actions
* Text extraction and summarization
* Automatic CSS selector detection

---

## **Features**

| Feature             | Description                                                    |
| ------------------- | -------------------------------------------------------------- |
| Multi-step tasks    | Execute sequences like: visit → search → click → summarize     |
| Search              | Automatic search for query on YouTube, Google, Wikipedia       |
| Summarization       | Extract text from page elements and summarize in 3-5 sentences |
| Screenshot          | Take screenshots of webpages or sections                       |
| Async browser       | Uses async Playwright for safe execution in Streamlit          |
| Automatic selectors | Click and summarize actions detect CSS selectors automatically |

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/SusmithaDasi/browser_mcp_agent
cd browser_mcp_agent
```

2. Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate   # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install Playwright browsers:

```bash
playwright install
```

5. Add your OpenAI API key in `.env`:

```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## **Run the App**

```bash
streamlit run main.py
```

Open the URL displayed in your terminal (usually `http://localhost:8501`) and start giving commands.

---

## **Example Commands**

* “Open YouTube and search for Bahubali”
* “Go to Wikipedia and search for Data Science, then summarize the first paragraph”
* “Visit Google and search for OpenAI GPT-4”
* “Go to openai.com, click the About link, scroll down, and take a screenshot”

---

## **Notes**

* The agent runs asynchronously, preserving browser state across multiple steps.
* Screenshots are saved in the project folder by default.
* Avoid giving commands that require login unless you add authentication handling.

---

## **Future Improvements**

* Automatically click the first search result after searching
* Extract video titles/descriptions from YouTube search results
* Task preview UI in Streamlit for multi-step actions
* Extend automatic search support to more websites (Amazon, Twitter, etc.)

---

## 👩‍💻 Author

**Susmitha Dasi**  
📍 Hyderabad, Telangana, India.
📧 todssusmitha@gmail.com

---

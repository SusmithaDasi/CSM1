import os
import json
from dotenv import load_dotenv
from browser_agent.playwright_wrapper import BrowserControllerAsync
from openai import OpenAI
import asyncio

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize async browser
browser = BrowserControllerAsync()
asyncio.run(browser.start())

def interpret_command(command):
    """
    Converts natural language commands into structured JSON instructions.
    Handles multi-step tasks, automatic selector detection, and search actions.
    """

    prompt = f"""
    You are a browser control agent. Split this natural language command into a JSON list of steps.
    Each step must have:
    - action: visit/click/scroll/screenshot/summarize/search
    - params: parameters (url, selector, amount, filename, query)

    Rules:
    - If the user wants to search on a website (YouTube, Google, Wikipedia), output action 'search' with 'url' and 'query'.
    - For click or summarize, detect CSS selectors automatically.
    - Respond ONLY with valid JSON list, no extra text.

    Command: "{command}"
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    content = response.choices[0].message.content.strip()

    # Remove Markdown code fences if present
    if content.startswith("```") and content.endswith("```"):
        content = "\n".join(content.splitlines()[1:-1]).strip()

    content = content.replace("```json", "").replace("```", "").strip()

    try:
        steps = json.loads(content)
    except json.JSONDecodeError:
        steps = [{"action": "error", "params": {"message": content}}]

    return steps

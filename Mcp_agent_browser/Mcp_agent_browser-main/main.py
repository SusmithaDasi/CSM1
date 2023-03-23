import streamlit as st
from browser_agent.agent import interpret_command, browser
import os
from openai import OpenAI
from dotenv import load_dotenv
import asyncio

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Browser MCP Agent", layout="wide")
st.title("🧠 Browser MCP Agent")
st.write("Control a browser using natural language commands.")

command = st.text_input("Enter your browsing command:")

async def run_steps(steps):
    for i, step in enumerate(steps, 1):
        action = step.get("action")
        params = step.get("params", {})

        st.write(f"**Step {i}: {action}**")

        if action == "visit":
            result = await browser.visit(params.get("url", ""))
            st.write(result)

        elif action == "click":
            result = await browser.click(params.get("selector", ""))
            st.write(result)

        elif action == "scroll":
            result = await browser.scroll(params.get("amount", 1000))
            st.write(result)

        elif action == "screenshot":
            file = await browser.screenshot(params.get("filename", "screenshot.png"))
            st.image(file)

        elif action == "summarize":
            selector = params.get("selector", "body")
            content = await browser.extract_text(selector)
            if content and content != "No element found with selector.":
                summary_prompt = f"Summarize this text in 3-5 sentences:\n{content}"
                summary = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.3
                )
                summary_text = summary.choices[0].message.content
                st.write(summary_text)
            else:
                st.write(content)

        elif action == "search":
            url = params.get("url", "")
            query = params.get("query", "")
            if url and query:
                await browser.visit(url)
                # Auto-detect input box based on site
                if "youtube.com" in url:
                    await browser.page.fill("input#search", query)
                    await browser.page.press("input#search", "Enter")
                    st.write(f"Searched YouTube for '{query}'")
                elif "google.com" in url:
                    await browser.page.fill("input[name='q']", query)
                    await browser.page.press("input[name='q']", "Enter")
                    st.write(f"Searched Google for '{query}'")
                elif "wikipedia.org" in url:
                    await browser.page.fill("input[name='search']", query)
                    await browser.page.press("input[name='search']", "Enter")
                    st.write(f"Searched Wikipedia for '{query}'")
                else:
                    st.warning(f"Search not supported automatically for {url}")
            else:
                st.error("Missing url or query for search action")

        else:
            st.error(f"Unknown action or error: {params.get('message', action)}")


if st.button("Run Command"):
    if not command.strip():
        st.warning("Please enter a command!")
    else:
        steps = interpret_command(command)
        asyncio.run(run_steps(steps))

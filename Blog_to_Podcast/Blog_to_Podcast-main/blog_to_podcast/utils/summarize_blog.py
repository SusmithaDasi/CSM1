from openai import OpenAI
import os

def summarize_blog(content: str) -> str:
    """
    Generates an engaging 2000-character summary using GPT-4.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key)

    prompt = (
        "You are a podcast script writer. Summarize the following blog engagingly "
        "in under 2000 characters, as if you're narrating it to a podcast audience:\n\n"
        f"{content}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

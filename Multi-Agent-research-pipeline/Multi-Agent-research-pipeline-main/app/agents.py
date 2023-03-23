import json

from langchain_ollama import ChatOllama

from app.state import ResearchState
from app.tools import search_web


llm = ChatOllama(model="llama3.2", temperature=0)


def clean_json_text(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    return text


def planner_agent(state: ResearchState) -> ResearchState:
    question = state["question"]

    prompt = f"""
You are a research planner.

Break the question into exactly 3 clear subtasks.

Question:
{question}

Return only valid JSON in this format:
{{
  "subtasks": [
    "subtask 1",
    "subtask 2",
    "subtask 3"
  ]
}}
"""

    response = llm.invoke(prompt)

    try:
        cleaned = clean_json_text(response.content)
        parsed = json.loads(cleaned)
        subtasks = parsed["subtasks"]
    except Exception:
        subtasks = [
            line.strip()
            for line in response.content.split("\n")
            if line.strip()
        ]

    subtasks = [
        task.strip()
        for task in subtasks
        if isinstance(task, str) and len(task.strip()) >= 2
    ]

    subtasks = subtasks[:3]

    return {
        **state,
        "subtasks": subtasks,
    }


def worker_agent(state: ResearchState) -> ResearchState:
    subtasks = state["subtasks"]
    critique = state.get("critique", "")
    outputs = {}

    for task in subtasks:
        search_results = search_web(task)

        prompt = f"""
You are a research worker.

Your subtask:
{task}

Previous critique, if any:
{critique}

Use these web search results:
{search_results}

Instructions:
- Fix any issues mentioned in the critique
- Answer the subtask clearly
- Only use information supported by the search results
- Mention important evidence
- Be concise
"""

        response = llm.invoke(prompt)

        outputs[task] = {
            "answer": response.content,
            "sources": search_results,
        }

    return {
    **state,
    "worker_outputs": outputs,
    "revision_count": state.get("revision_count", 0) + 1,
    }


def critic_agent(state: ResearchState) -> ResearchState:
    worker_outputs = state["worker_outputs"]

    prompt = f"""
You are a strict research critic.

Review these worker outputs:
{worker_outputs}

Decide:
1. Are they sufficient? yes/no
2. Are claims supported by sources?
3. Are there missing details, vague claims, or unsupported claims?

Return only valid JSON in this format:
{{
  "needs_revision": false,
  "critique": "your explanation"
}}
"""

    response = llm.invoke(prompt)

    try:
        cleaned = clean_json_text(response.content)
        parsed = json.loads(cleaned)
        needs_revision = bool(parsed["needs_revision"])
        critique = parsed["critique"]
    except Exception:
        needs_revision = False
        critique = response.content

    return {
        **state,
        "critique": critique,
        "needs_revision": needs_revision,
    }


def final_agent(state: ResearchState) -> ResearchState:
    question = state["question"]
    worker_outputs = state["worker_outputs"]
    critique = state["critique"]

    prompt = f"""
You are a professional research analyst.

Original question:
{question}

Worker outputs:
{worker_outputs}

Critic review:
{critique}

Write a structured final report with:

1. Answer (clear, direct)
2. Key Insights (bullet points)
3. Limitations (what might be missing or uncertain)
4. Sources (list important sources)

Keep it clean and readable.
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "final_answer": response.content,
    }
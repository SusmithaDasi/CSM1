# Multi-Agent Research Pipeline

A prototype research assistant that coordinates multiple specialized agents to answer user questions.

The pipeline uses:
- `planner_agent` to split a question into a fixed set of subtasks
- `worker_agent` to research each subtask via web search
- `critic_agent` to evaluate the worker outputs and decide whether revision is needed
- `final_agent` to synthesize a structured final report

It includes:
- `app/main.py` — simple CLI entrypoint
- `app/api.py` — FastAPI backend exposing a `/research` endpoint
- `app/ui.py` — Streamlit frontend for interactive use
- `app/graph.py` — LangGraph state graph orchestration
- `app/agents.py` — agent implementations and LLM prompts
- `app/tools.py` — web search integration via Tavily
- `app/state.py` — typed research state contract

## Architecture

The research workflow is defined in `app/graph.py`:
1. `planner` creates 3 subtasks from the user question
2. `worker` searches and answers each subtask
3. `critic` checks whether outputs need revision
4. loop back to `worker` if revision is required (up to 2 times)
5. `final` writes a polished final report

The graph is compiled into an executable state machine using `StateGraph`.

## Requirements

The project depends on:
- Python 3.11+ (recommended)
- `fastapi`
- `uvicorn`
- `streamlit`
- `requests`
- `python-dotenv`
- `tavily`
- `langchain_ollama`
- `langgraph`

If you use the included `.venv`, activate it before installing or running the app.

## Environment

The app loads environment variables from `.env` using `python-dotenv`.

Create a `.env` file with your Tavily credentials or other required API keys. Example:

```env
TAVILY_API_KEY=your_api_key_here
```

## Run the backend API

Start the FastAPI server:

```bash
uvicorn app.api:app --reload
```

Then the REST API is available at `http://127.0.0.1:8000/research`.

## Run the Streamlit UI

Start the UI in a separate terminal after the backend is running:

```bash
streamlit run app/ui.py
```

Open the browser link shown by Streamlit and enter a research question.

## Run the CLI mode

Use the CLI entrypoint for a terminal-based workflow:

```bash
python app/main.py
```

## Key files

- `app/agents.py` — defines the planner, worker, critic, and final agents
- `app/tools.py` — performs web searches using Tavily
- `app/graph.py` — builds the orchestrated state graph
- `app/api.py` — exposes the research pipeline as a REST API
- `app/ui.py` — Streamlit interface that calls the API
- `app/state.py` — typed `ResearchState` payload structure

## Notes

- `app/ui.py` requires the FastAPI backend to be running.
- The worker agent uses web search results from Tavily to ground its answers.
- The critic agent can force up to two revisions before final synthesis.

## Troubleshooting

- If `streamlit` shows a connection error, confirm `uvicorn` is running and the API URL is correct.
- If the pipeline cannot search, confirm your `.env` keys are loaded and Tavily credentials are valid.

---

## 👩‍💻 Author

**Susmitha Dasi**  
📍 Hyderabad, Telangana, India.
📧 todssusmitha@gmail.com

---


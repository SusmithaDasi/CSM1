from fastapi import FastAPI
from pydantic import BaseModel

from app.graph import build_graph


app = FastAPI(title="Multi-Agent Research Pipeline")

graph = build_graph()


class ResearchRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Multi-Agent Research Pipeline API is running"
    }


@app.post("/research")
def research(request: ResearchRequest):
    initial_state = {
        "question": request.question,
        "subtasks": [],
        "worker_outputs": {},
        "critique": "",
        "final_answer": "",
        "needs_revision": False,
        "revision_count": 0,
    }

    result = graph.invoke(initial_state)

    return {
        "question": result["question"],
        "subtasks": result["subtasks"],
        "worker_outputs": result["worker_outputs"],
        "critique": result["critique"],
        "final_answer": result["final_answer"],
        "revision_count": result["revision_count"],
    }
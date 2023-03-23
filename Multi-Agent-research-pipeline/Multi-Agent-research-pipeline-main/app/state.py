from typing import TypedDict, List, Dict


class ResearchState(TypedDict):
    question: str
    subtasks: List[str]
    worker_outputs: Dict[str, dict]
    critique: str
    final_answer: str
    needs_revision: bool
    revision_count: int
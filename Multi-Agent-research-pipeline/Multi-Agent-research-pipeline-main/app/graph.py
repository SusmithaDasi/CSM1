from langgraph.graph import StateGraph, END

from app.state import ResearchState
from app.agents import (
    planner_agent,
    worker_agent,
    critic_agent,
    final_agent,
)


def should_revise(state: ResearchState):
    if state.get("needs_revision") and state.get("revision_count", 0) < 2:
        return "worker"

    return "final"


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_agent)
    graph.add_node("worker", worker_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("final", final_agent)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "worker")
    graph.add_edge("worker", "critic")

    graph.add_conditional_edges(
        "critic",
        should_revise,
        {
            "worker": "worker",
            "final": "final",
        },
    )

    graph.add_edge("final", END)

    return graph.compile()
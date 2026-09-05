"""
ScopeOut — Graph Definition (Phase 3)
======================================
Fan-out pipeline using LangGraph's Send() API:

  START → planner → [worker, worker, worker, worker] → synthesizer → END
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     These run in PARALLEL — one per angle

The planner's outgoing edge is a function that returns Send() objects.
Each Send spins up a research_worker with one angle. LangGraph runs
them all concurrently, merges findings via operator.add, and only
proceeds to the synthesizer once every worker is done.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from nodes import planner, research_worker, synthesizer
from state import ResearchAngle, ScopeOutState


def route_to_workers(state: ScopeOutState) -> list[Send]:
    """
    Fan-out: dispatch one parallel worker per research angle.

    Called automatically after the planner finishes. Returns a list
    of Send objects — LangGraph launches them all at once.
    """
    return [
        Send("research_worker", {
            "company": state["company"],
            "angle": angle,
        })
        for angle in state["angles"]
    ]


def build_graph():
    """Build and compile the ScopeOut graph with parallel workers."""

    workflow = StateGraph(ScopeOutState)

    # ── Add nodes ───────────────────────────────────────
    workflow.add_node("planner", planner)
    workflow.add_node("research_worker", research_worker)
    workflow.add_node("synthesizer", synthesizer)

    # ── Wire edges ──────────────────────────────────────
    workflow.add_edge(START, "planner")

    # Conditional edge: planner fans out to parallel workers
    workflow.add_conditional_edges("planner", route_to_workers, ["research_worker"])

    # All worker instances converge here — synthesizer waits
    # until every worker is done before running
    workflow.add_edge("research_worker", "synthesizer")

    workflow.add_edge("synthesizer", END)

    return workflow.compile()
"""
ScopeOut — Graph Definition (Phase 1)
======================================
A linear 3-node pipeline:

  START → planner → researcher → synthesizer → END

Phase 3 will introduce parallel worker branches.
Phase 4 will add a conditional critic loop.
"""

from langgraph.graph import END, START, StateGraph

from nodes import planner, researcher, synthesizer
from state import ScopeOutState


def build_graph():
    """Build and compile the ScopeOut graph."""

    workflow = StateGraph(ScopeOutState)

    # ── Add nodes ───────────────────────────────────────
    workflow.add_node("planner", planner)
    workflow.add_node("researcher", researcher)
    workflow.add_node("synthesizer", synthesizer)

    # ── Wire edges (linear for Phase 1) ─────────────────
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()

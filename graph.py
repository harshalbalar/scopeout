"""
ScopeOut — Graph Definition (Phase 4)
======================================
Fan-out pipeline with a critic quality loop:

  START -> planner -> [workers] -> critic -> synthesizer -> END
                         ^           |
                         |___redo____|

The conditional edge after the critic either:
  - Routes to synthesizer (all findings pass, or max retries reached)
  - Sends flagged angles back to workers for a redo via Send()

Max 1 redo round to prevent infinite loops.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from nodes import critic, planner, research_worker, synthesizer
from state import ScopeOutState


# ── Routing Functions ───────────────────────────────────────────────


def route_to_workers(state: ScopeOutState) -> list[Send]:
    """
    Fan-out: dispatch one parallel worker per research angle.
    Called after the planner finishes.
    """
    return [
        Send("research_worker", {
            "company": state["company"],
            "angle": angle,
        })
        for angle in state["angles"]
    ]


def route_after_critic(state: ScopeOutState) -> list[Send] | str:
    """
    Decide what happens after the critic reviews findings.

    If all pass (or we've hit the retry cap): proceed to synthesizer.
    If some are flagged: send ONLY those angles back to workers,
    with the critic's specific feedback so the worker knows what
    to improve.
    """
    flagged = state.get("flagged_topics", [])
    retry_count = state.get("retry_count", 0)

    # Accept what we have if all pass or we've already retried once
    if not flagged or retry_count >= 2:
        return "synthesizer"

    # Build per-topic feedback from the critique
    critique = state.get("critique", [])
    feedback_by_topic = {
        item["topic"]: item.get("reason", "Needs improvement")
        for item in critique
        if item.get("verdict") == "redo"
    }

    # Send only the flagged angles back — unflagged findings stay as-is
    return [
        Send("research_worker", {
            "company": state["company"],
            "angle": angle,
            "feedback": feedback_by_topic.get(angle.topic, "Needs improvement"),
        })
        for angle in state["angles"]
        if angle.topic in flagged
    ]


# ── Graph Construction ──────────────────────────────────────────────


def build_graph():
    """Build and compile the ScopeOut graph with parallel workers + critic loop."""

    workflow = StateGraph(ScopeOutState)

    # ── Add nodes ───────────────────────────────────────
    workflow.add_node("planner", planner)
    workflow.add_node("research_worker", research_worker)
    workflow.add_node("critic", critic)
    workflow.add_node("synthesizer", synthesizer)

    # ── Wire edges ──────────────────────────────────────
    workflow.add_edge(START, "planner")

    # Planner fans out to parallel workers
    workflow.add_conditional_edges("planner", route_to_workers, ["research_worker"])

    # Workers converge into the critic
    workflow.add_edge("research_worker", "critic")

    # Critic either approves (-> synthesizer) or flags (-> workers redo)
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        ["synthesizer", "research_worker"],
    )

    workflow.add_edge("synthesizer", END)

    return workflow.compile()
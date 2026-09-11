"""
ScopeOut — Graph Definition (Optimized)
=========================================
Optimized flow:
  START → planner → batch_search → [analyze_workers × 4] → critic → synthesizer → END
                                                               ↕
                                                        [research_worker redo]

batch_search fires all Tavily searches concurrently in one node.
analyze_workers do Gemini analysis only (reading pre-fetched results).
research_worker handles redos (full search + analyze).
synthesizer streams tokens to the browser.
"""
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from nodes import planner, batch_search, analyze_worker, research_worker, critic, synthesizer
from state import ScopeOutState


def route_to_analyzers(state: ScopeOutState) -> list[Send]:
    """Fan out: one analyze_worker per angle, with pre-fetched search results."""
    search_results = state.get("search_results", {})
    return [
        Send("analyze_worker", {
            "company": state["company"],
            "angle": angle,
            "search_hits": search_results.get(angle.topic, []),
        })
        for angle in state["angles"]
    ]


def route_after_critic(state: ScopeOutState) -> list[Send] | str:
    flagged = state.get("flagged_topics", [])
    retry_count = state.get("retry_count", 0)
    if not flagged or retry_count >= 2:
        return "synthesizer"
    critique = state.get("critique", [])
    feedback_by_topic = {
        item["topic"]: item.get("reason", "Needs improvement")
        for item in critique if item.get("verdict") == "redo"
    }
    return [
        Send("research_worker", {
            "company": state["company"],
            "angle": angle,
            "feedback": feedback_by_topic.get(angle.topic, "Needs improvement"),
        })
        for angle in state["angles"]
        if angle.topic in flagged
    ]


def build_graph():
    workflow = StateGraph(ScopeOutState)

    workflow.add_node("planner", planner)
    workflow.add_node("batch_search", batch_search)
    workflow.add_node("analyze_worker", analyze_worker)
    workflow.add_node("research_worker", research_worker)  # for redos
    workflow.add_node("critic", critic)
    workflow.add_node("synthesizer", synthesizer)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "batch_search")
    workflow.add_conditional_edges("batch_search", route_to_analyzers, ["analyze_worker"])
    workflow.add_edge("analyze_worker", "critic")
    workflow.add_conditional_edges("critic", route_after_critic, ["synthesizer", "research_worker"])
    workflow.add_edge("research_worker", "critic")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()
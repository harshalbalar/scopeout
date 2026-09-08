"""
ScopeOut — Graph Nodes (Phase 5)
=================================
Phase 5 upgrade: every node now emits structured events to the
EventBus so the FastAPI server can stream them to the frontend.
Print statements kept for CLI usage via main.py.
"""

from __future__ import annotations

import json
import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

from event_bus import bus
from state import ResearchAngle, ResearchFinding, ScopeOutState


# ── Shared clients ──────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)
tavily_client = TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"],
)


# ── Helpers ─────────────────────────────────────────────────────────


def _parse_json(text: str):
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


# ── Nodes ───────────────────────────────────────────────────────────


def planner(state: ScopeOutState) -> dict:
    company = state["company"]

    bus.emit("planner_start", company=company)
    print(f"\n[planner] Planning research for '{company}'...")

    prompt = f"""You are a competitive intelligence analyst planning a teardown of "{company}".

Identify exactly 4 research angles. For each, provide:
- "topic": a short label (use exactly these four: "Pricing & Plans", "Key Features", "Customer Sentiment", "Market Positioning")
- "question": a specific, actionable research question tailored to {company}

Respond with ONLY a JSON array of 4 objects. No other text, no markdown fences."""

    response = llm.invoke(prompt)
    angles_data = _parse_json(response.content)
    angles = [ResearchAngle(**angle) for angle in angles_data]

    print(f"[planner] Identified {len(angles)} research angles")
    for a in angles:
        print(f"  -> {a.topic}: {a.question}")

    bus.emit("planner_done", angles=[a.model_dump() for a in angles])

    return {"angles": angles}


def research_worker(state: dict) -> dict:
    company = state["company"]
    angle = state["angle"]
    feedback = state.get("feedback", "")

    if isinstance(angle, dict):
        angle = ResearchAngle(**angle)

    is_redo = bool(feedback)

    # ── Step 1: Search ──────────────────────────────────
    if is_redo:
        query = f"{company} {angle.question}"
        bus.emit("worker_start", topic=angle.topic, query=query, is_redo=True,
                 feedback=feedback)
        print(f"\n[worker:{angle.topic}] REDO — Searching with refined query")
    else:
        query = f"{company} {angle.topic}"
        bus.emit("worker_start", topic=angle.topic, query=query, is_redo=False)
        print(f"\n[worker:{angle.topic}] Searching: '{query}'")

    search_response = tavily_client.search(
        query=query,
        max_results=7 if is_redo else 5,
        search_depth="advanced",
    )

    results = search_response.get("results", [])

    # ── Step 2: Analyze ─────────────────────────────────
    if results:
        context = "\n\n".join(
            f"[Source: {r['url']}]\nTitle: {r.get('title', 'N/A')}\n{r.get('content', '')}"
            for r in results
        )
        sources = [r["url"] for r in results]

        redo_instruction = ""
        if is_redo:
            redo_instruction = f"""
IMPORTANT — PREVIOUS ATTEMPT WAS FLAGGED:
{feedback}
Address this issue specifically in your revised analysis."""

        prompt = f"""You are a competitive intelligence researcher analyzing "{company}".

Research angle: {angle.topic}
Question: {angle.question}
{redo_instruction}

Below are web search results. Analyze them to answer the research question.
Be specific: cite actual numbers, plan names, feature details, and concrete facts.
If the search results conflict, note the discrepancy.
Only include information that is supported by the search results below.

SEARCH RESULTS:
{context}

Provide a detailed, well-structured analysis in under 400 words."""

    else:
        print(f"[worker:{angle.topic}] No search results, falling back to LLM knowledge")
        sources = []
        prompt = f"""You are a competitive intelligence researcher analyzing "{company}".

Research angle: {angle.topic}
Question: {angle.question}

No web search results were available. Provide your best analysis based on
your training knowledge. Keep your response under 300 words."""

    response = llm.invoke(prompt)

    finding = ResearchFinding(
        topic=angle.topic,
        content=response.content,
        sources=sources,
    )

    tag = "REDO done" if is_redo else "Done"
    print(f"[worker:{angle.topic}] {tag} ({len(sources)} sources)")
    bus.emit("worker_done", topic=angle.topic, sources_count=len(sources),
             is_redo=is_redo, sources=sources)

    return {"findings": [finding]}


def critic(state: ScopeOutState) -> dict:
    findings = state["findings"]
    retry_count = state.get("retry_count", 0)

    bus.emit("critic_start", round=retry_count + 1)
    print(f"\n[critic] Evaluating {len(findings)} findings (round {retry_count + 1})...")

    # ── Programmatic pre-check ──────────────────────────
    auto_flagged = set()
    for f in findings:
        if len(f.sources) < 2:
            auto_flagged.add(f.topic)

    # ── LLM evaluation ─────────────────────────────────
    findings_text = "\n\n---\n\n".join(
        f"Topic: {f.topic}\nWord count: {len(f.content.split())}\n"
        f"Number of sources: {len(f.sources)}\n"
        f"Content preview: {f.content[:500]}..."
        for f in findings
    )

    prompt = f"""You are a research quality critic for competitive intelligence reports.

Evaluate each research finding below on three criteria:

1. SOURCING: Does it appear to draw from at least 2 distinct, credible sources?
2. SPECIFICITY: Does it include concrete details — actual prices, plan names,
   feature names, percentages, competitor names?
3. DEPTH: Does the analysis have enough substance (150+ words) to be useful?

A finding must pass ALL THREE to get a "pass" verdict.

FINDINGS:
{findings_text}

For each finding, return:
- "topic": the exact topic label
- "verdict": "pass" or "redo"
- "reason": if "redo", a specific 1-sentence explanation. if "pass", empty string.

Return ONLY a JSON array. No other text, no markdown fences."""

    response = llm.invoke(prompt)
    try:
        evaluations = _parse_json(response.content)
    except Exception:
        # Local model returned bad JSON — auto-pass all findings
        print("[critic] JSON parse failed, auto-passing all findings")
        evaluations = [{"topic": f.topic, "verdict": "pass", "reason": ""} for f in findings]

    flagged_topics = set(auto_flagged)
    for ev in evaluations:
        if ev.get("verdict") == "redo":
            flagged_topics.add(ev["topic"])

    for ev in evaluations:
        topic = ev["topic"]
        is_flagged = topic in flagged_topics
        verdict = "redo" if is_flagged else "pass"
        reason = ev.get("reason", "")
        if is_flagged and not reason:
            reason = "insufficient sources"

        print(f"[critic] {topic}: {verdict.upper()}" + (f" — {reason}" if reason else ""))
        bus.emit("critic_verdict", topic=topic, verdict=verdict, reason=reason)

    if flagged_topics:
        print(f"[critic] {len(flagged_topics)} topic(s) flagged for redo")
    else:
        print("[critic] All findings passed quality review")

    bus.emit("critic_done", flagged=list(flagged_topics), round=retry_count + 1)

    return {
        "critique": evaluations,
        "flagged_topics": list(flagged_topics),
        "retry_count": retry_count + 1,
    }


def synthesizer(state: ScopeOutState) -> dict:
    company = state["company"]
    findings = state["findings"]

    bus.emit("synthesizer_start")
    print(f"\n[synthesizer] Writing report...")

    findings_text = "\n\n---\n\n".join(
        f"### {f.topic}\n{f.content}\n\nSources: {', '.join(f.sources) if f.sources else 'LLM knowledge'}"
        for f in findings
    )

    prompt = f"""You are a senior competitive intelligence analyst. Synthesize the research
below into a polished competitive teardown report for "{company}".

RESEARCH FINDINGS:
{findings_text}

FORMAT:
- **Executive Summary** (2-3 sentences)
- **Pricing & Plans**
- **Key Features & Differentiators**
- **Customer Sentiment**
- **Market Positioning**
- **Bottom Line** (strengths, vulnerabilities, opportunities)

RULES:
- Use markdown formatting.
- After each major factual claim, add a source reference as a markdown link.
- Be direct and analytical — professional teardown, not marketing copy.
- Every factual claim must be traceable to the provided sources."""

    response = llm.invoke(prompt)

    print(f"[synthesizer] Report generated ({len(response.content)} chars)")
    bus.emit("synthesizer_done", report_length=len(response.content))

    return {"report": response.content}
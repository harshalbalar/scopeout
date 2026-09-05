"""
ScopeOut — Graph Nodes (Phase 3)
=================================
Phase 3 upgrade: the single sequential researcher is replaced by
parallel research workers using LangGraph's Send() API.

  planner → [worker, worker, worker, worker] → synthesizer

Each worker handles one research angle independently. They all run
at the same time and their findings merge via the operator.add
reducer on the state's findings list.
"""

from __future__ import annotations

import json
import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

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
    """
    Extract JSON from an LLM response.

    Gemini sometimes wraps JSON in ```json ... ``` fences.
    This strips those before parsing.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


# ── Nodes ───────────────────────────────────────────────────────────


def planner(state: ScopeOutState) -> dict:
    """
    Break the company/product into concrete research angles.

    Returns 4 angles: pricing, features, customer reviews, and
    market positioning — each with a tailored research question.
    """
    company = state["company"]

    prompt = f"""You are a competitive intelligence analyst planning a teardown of "{company}".

Identify exactly 4 research angles. For each, provide:
- "topic": a short label (use exactly these four: "Pricing & Plans", "Key Features", "Customer Sentiment", "Market Positioning")
- "question": a specific, actionable research question tailored to {company}

Respond with ONLY a JSON array of 4 objects. No other text, no markdown fences."""

    response = llm.invoke(prompt)
    angles_data = _parse_json(response.content)
    angles = [ResearchAngle(**angle) for angle in angles_data]

    print(f"[planner] Identified {len(angles)} research angles for '{company}'")
    for a in angles:
        print(f"  -> {a.topic}: {a.question}")

    return {"angles": angles}


def research_worker(state: dict) -> dict:
    """
    Research a SINGLE angle using Tavily web search + Gemini analysis.

    This node is instantiated in parallel by the Send() API — one
    instance per research angle, all running at the same time.

    Receives from Send():
        {"company": str, "angle": ResearchAngle}

    Returns:
        {"findings": [ResearchFinding]}
        The operator.add reducer merges findings from all workers.
    """
    company = state["company"]
    angle = state["angle"]

    # Handle case where Send() serialized the Pydantic model to a dict
    if isinstance(angle, dict):
        angle = ResearchAngle(**angle)

    # ── Step 1: Search the web via Tavily ───────────────
    query = f"{company} {angle.topic}"
    print(f"[worker:{angle.topic}] Searching: '{query}'")

    search_response = tavily_client.search(
        query=query,
        max_results=5,
        search_depth="advanced",
    )

    results = search_response.get("results", [])

    # ── Step 2: Build context and analyze ───────────────
    if results:
        context = "\n\n".join(
            f"[Source: {r['url']}]\nTitle: {r.get('title', 'N/A')}\n{r.get('content', '')}"
            for r in results
        )
        sources = [r["url"] for r in results]

        prompt = f"""You are a competitive intelligence researcher analyzing "{company}".

Research angle: {angle.topic}
Question: {angle.question}

Below are web search results. Analyze them to answer the research question.
Be specific: cite actual numbers, plan names, feature details, and concrete facts.
If the search results conflict, note the discrepancy.
Only include information that is supported by the search results below.

SEARCH RESULTS:
{context}

Provide a detailed, well-structured analysis in under 400 words."""

    else:
        # Fallback: no search results — use LLM knowledge
        print(f"[worker:{angle.topic}] No search results, falling back to LLM knowledge")
        sources = []

        prompt = f"""You are a competitive intelligence researcher analyzing "{company}".

Research angle: {angle.topic}
Question: {angle.question}

No web search results were available. Provide your best analysis based on
your training knowledge. Be specific where possible, but note that this is
based on training data and may not reflect the latest information.

Keep your response under 300 words."""

    response = llm.invoke(prompt)

    finding = ResearchFinding(
        topic=angle.topic,
        content=response.content,
        sources=sources,
    )

    print(f"[worker:{angle.topic}] Done ({len(sources)} sources)")

    return {"findings": [finding]}


def synthesizer(state: ScopeOutState) -> dict:
    """
    Combine all research findings into a polished, sourced teardown report.
    """
    company = state["company"]
    findings = state["findings"]

    findings_text = "\n\n---\n\n".join(
        f"### {f.topic}\n{f.content}\n\nSources: {', '.join(f.sources) if f.sources else 'LLM knowledge (no web sources)'}"
        for f in findings
    )

    prompt = f"""You are a senior competitive intelligence analyst. Synthesize the research
below into a polished competitive teardown report for "{company}".

RESEARCH FINDINGS:
{findings_text}

FORMAT:
- **Executive Summary** (2-3 sentences: what {company} is, who it's for, its market position)
- **Pricing & Plans** (specific numbers and plan names)
- **Key Features & Differentiators** (what sets it apart)
- **Customer Sentiment** (what users love and complain about)
- **Market Positioning** (competitors, where {company} fits)
- **Bottom Line** (strategic takeaways: strengths, vulnerabilities, opportunities)

RULES:
- Use markdown formatting.
- After each major factual claim, add a source reference as a markdown link, e.g. [source](url).
- Be direct and analytical — this is a professional teardown, not marketing copy.
- Every factual claim must be traceable to the provided sources."""

    response = llm.invoke(prompt)

    print(f"[synthesizer] Report generated ({len(response.content)} chars)")

    return {"report": response.content}
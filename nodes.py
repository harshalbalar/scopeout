"""
ScopeOut — Graph Nodes (Phase 4)
=================================
Phase 4 upgrade: added critic node that reviews findings quality
and routes weak ones back to workers for a redo.

  planner -> [workers] -> critic -> synthesizer
                 ^           |
                 |___redo____|

The worker also now accepts optional feedback from the critic,
using it to adjust search queries and analysis on redo attempts.
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
    Strips markdown code fences if present.
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

    print(f"\n[planner] Identified {len(angles)} research angles for '{company}'")
    for a in angles:
        print(f"  -> {a.topic}: {a.question}")

    return {"angles": angles}


def research_worker(state: dict) -> dict:
    """
    Research a SINGLE angle using Tavily web search + Gemini analysis.

    Instantiated in parallel via Send(). On a redo attempt, the worker
    receives critic feedback and uses it to adjust the search query
    and analysis prompt for a better result.

    Receives from Send():
        {"company": str, "angle": ResearchAngle, "feedback": str (optional)}
    """
    company = state["company"]
    angle = state["angle"]
    feedback = state.get("feedback", "")

    if isinstance(angle, dict):
        angle = ResearchAngle(**angle)

    is_redo = bool(feedback)

    # ── Step 1: Search the web ──────────────────────────
    # On redo, use the full research question for a more targeted search
    if is_redo:
        query = f"{company} {angle.question}"
        print(f"\n[worker:{angle.topic}] REDO — Searching with refined query")
        print(f"  Feedback: {feedback}")
    else:
        query = f"{company} {angle.topic}"
        print(f"\n[worker:{angle.topic}] Searching: '{query}'")

    search_response = tavily_client.search(
        query=query,
        max_results=7 if is_redo else 5,
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

        redo_instruction = ""
        if is_redo:
            redo_instruction = f"""
IMPORTANT — PREVIOUS ATTEMPT WAS FLAGGED:
{feedback}
Address this issue specifically in your revised analysis. Provide more concrete
details, cite more sources, and ensure depth of coverage."""

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
your training knowledge. Be specific where possible, but note that this is
based on training data and may not reflect the latest information.

Keep your response under 300 words."""

    response = llm.invoke(prompt)

    finding = ResearchFinding(
        topic=angle.topic,
        content=response.content,
        sources=sources,
    )

    tag = "REDO done" if is_redo else "Done"
    print(f"[worker:{angle.topic}] {tag} ({len(sources)} sources)")

    return {"findings": [finding]}


def critic(state: ScopeOutState) -> dict:
    """
    Review all research findings for quality, sourcing, and depth.

    Checks each finding on three criteria:
      1. SOURCING  — at least 2 real, distinct sources
      2. SPECIFICITY — concrete numbers, names, prices, not vague claims
      3. DEPTH — substantively answers the research question

    Returns structured evaluations and a list of flagged topics.
    A programmatic pre-check catches zero-source findings automatically.
    """
    findings = state["findings"]
    retry_count = state.get("retry_count", 0)

    print(f"\n[critic] Evaluating {len(findings)} findings (round {retry_count + 1})...")

    # ── Programmatic pre-check ──────────────────────────
    auto_flagged = set()
    for f in findings:
        if len(f.sources) < 2:
            auto_flagged.add(f.topic)
            print(f"[critic] {f.topic}: auto-flagged (only {len(f.sources)} source(s))")

    # ── LLM evaluation ─────────────────────────────────
    findings_text = "\n\n---\n\n".join(
        f"Topic: {f.topic}\n"
        f"Word count: {len(f.content.split())}\n"
        f"Number of sources: {len(f.sources)}\n"
        f"Content preview: {f.content[:500]}..."
        for f in findings
    )

    prompt = f"""You are a research quality critic for competitive intelligence reports.

Evaluate each research finding below on three criteria:

1. SOURCING: Does it appear to draw from at least 2 distinct, credible sources?
   Zero or one source is an automatic failure.
2. SPECIFICITY: Does it include concrete details — actual prices, plan names,
   feature names, percentages, competitor names? Vague generalizations fail.
3. DEPTH: Does the analysis have enough substance (150+ words of real analysis)
   to be useful in a professional competitive teardown? Thin summaries fail.

A finding must pass ALL THREE to get a "pass" verdict.

FINDINGS:
{findings_text}

For each finding, return:
- "topic": the exact topic label
- "verdict": "pass" or "redo"
- "reason": if "redo", a specific 1-sentence explanation of what is weak or missing.
            if "pass", set to empty string.

Return ONLY a JSON array. No other text, no markdown fences."""

    response = llm.invoke(prompt)
    evaluations = _parse_json(response.content)

    # ── Combine programmatic + LLM flags ────────────────
    flagged_topics = set(auto_flagged)
    for ev in evaluations:
        if ev.get("verdict") == "redo":
            flagged_topics.add(ev["topic"])

    # ── Print results ───────────────────────────────────
    for ev in evaluations:
        topic = ev["topic"]
        is_flagged = topic in flagged_topics
        status = "REDO" if is_flagged else "PASS"
        reason = ev.get("reason", "")

        if is_flagged and not reason:
            reason = "insufficient sources" if topic in auto_flagged else "flagged by review"

        line = f"[critic] {topic}: {status}"
        if reason:
            line += f" — {reason}"
        print(line)

    if flagged_topics:
        print(f"[critic] {len(flagged_topics)} topic(s) flagged for redo")
    else:
        print("[critic] All findings passed quality review")

    return {
        "critique": evaluations,
        "flagged_topics": list(flagged_topics),
        "retry_count": retry_count + 1,
    }


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

    print(f"\n[synthesizer] Report generated ({len(response.content)} chars)")

    return {"report": response.content}
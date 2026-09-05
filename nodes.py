"""
ScopeOut — Graph Nodes (Phase 1)
=================================
Three nodes forming a linear pipeline:
  planner  →  researcher  →  synthesizer

The researcher uses LLM knowledge only (no web search until Phase 2).
"""

from __future__ import annotations

import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI

from state import ResearchAngle, ResearchFinding, ScopeOutState


# ── Shared LLM instance ────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,  # keep structured output deterministic
)


# ── Helpers ─────────────────────────────────────────────────────────


def _parse_json(text: str):
    """
    Extract JSON from an LLM response.

    Gemini sometimes wraps JSON in ```json ... ``` fences.
    This strips those before parsing.
    """
    text = text.strip()
    # Remove markdown code fences if present
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
        print(f"  → {a.topic}: {a.question}")

    return {"angles": angles}


def researcher(state: ScopeOutState) -> dict:
    """
    Investigate each research angle using the LLM's own knowledge.

    Phase 1: no web search — findings come from model training data.
    Phase 2 will swap this for Tavily-backed research.

    Returns all findings at once. The operator.add reducer on the state
    will append them to the findings list (matters in Phase 3 when
    multiple workers each return their own finding).
    """
    company = state["company"]
    angles = state["angles"]
    findings: list[ResearchFinding] = []

    for angle in angles:
        prompt = f"""You are a competitive intelligence researcher analyzing "{company}".

Research angle: {angle.topic}
Question: {angle.question}

Provide a detailed, factual analysis. Be specific: name actual plans and prices,
real features, known competitors, concrete details — not vague generalities.

Important: your knowledge has a training cutoff, so note where information might
be outdated. Keep your response focused and under 300 words."""

        response = llm.invoke(prompt)

        findings.append(
            ResearchFinding(
                topic=angle.topic,
                content=response.content,
                sources=[],  # No sources until Phase 2
            )
        )
        print(f"[researcher] Completed: {angle.topic}")

    return {"findings": findings}


def synthesizer(state: ScopeOutState) -> dict:
    """
    Combine all research findings into a polished competitive teardown.
    """
    company = state["company"]
    findings = state["findings"]

    findings_text = "\n\n---\n\n".join(
        f"### {f.topic}\n{f.content}" for f in findings
    )

    prompt = f"""You are a senior competitive intelligence analyst. Synthesize the research
below into a polished competitive teardown report for "{company}".

RESEARCH FINDINGS:
{findings_text}

FORMAT:
- **Executive Summary** (2–3 sentences: what {company} is, who it's for, its market position)
- **Pricing & Plans** (specifics, not vague)
- **Key Features & Differentiators** (what sets it apart)
- **Customer Sentiment** (what users love and complain about)
- **Market Positioning** (competitors, where {company} fits)
- **Bottom Line** (strategic takeaways: strengths, vulnerabilities, opportunities)

Use markdown. Be direct and analytical — this is a professional teardown, not marketing copy.
Note: this analysis is based on training data and may not reflect the very latest changes."""

    response = llm.invoke(prompt)

    print(f"[synthesizer] Report generated ({len(response.content)} chars)")

    return {"report": response.content}

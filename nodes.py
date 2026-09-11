"""
ScopeOut — Graph Nodes (Optimized)
====================================
Two key optimizations:
1. batch_search: all 4 Tavily searches fire concurrently via ThreadPoolExecutor
2. synthesizer: streams report tokens to the browser in real time

Flow: planner → batch_search → [analyze_workers] → critic → synthesizer (streaming)
Redos use full research_worker (search + analyze in one).
"""
from __future__ import annotations
import json, os, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from event_bus import bus
from state import ResearchAngle, ResearchFinding, ScopeOutState

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def _parse_json(text):
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match: text = match.group(1).strip()
    return json.loads(text)


# ── Planner ─────────────────────────────────────────

def planner(state: ScopeOutState) -> dict:
    company = state["company"]
    bus.emit("planner_start", company=company)
    print(f"\n[planner] Planning for '{company}'...")

    prompt = f"""You are a competitive intelligence analyst planning a teardown of "{company}".
Identify exactly 4 research angles. For each, provide:
- "topic": a short label (use exactly: "Pricing & Plans", "Key Features", "Customer Sentiment", "Market Positioning")
- "question": a specific research question tailored to {company}
Respond with ONLY a JSON array of 4 objects. No other text."""

    response = llm.invoke(prompt)
    angles = [ResearchAngle(**a) for a in _parse_json(response.content)]
    print(f"[planner] {len(angles)} angles identified")
    bus.emit("planner_done", angles=[a.model_dump() for a in angles])
    return {"angles": angles}


# ── Batch Search (concurrent Tavily) ────────────────

def batch_search(state: ScopeOutState) -> dict:
    """
    Fire all 4 Tavily searches concurrently using ThreadPoolExecutor.
    This completes all web searches in ~2-3s wall time instead of
    being interleaved with Gemini calls.
    """
    company = state["company"]
    angles = state["angles"]

    print(f"\n[batch_search] Searching all {len(angles)} angles concurrently...")

    def search_one(angle):
        query = f"{company} {angle.topic}"
        bus.emit("worker_start", topic=angle.topic, query=query, is_redo=False)
        print(f"[search] {angle.topic}: '{query}'")
        result = tavily_client.search(query=query, max_results=5, search_depth="basic")
        hits = result.get("results", [])
        print(f"[search] {angle.topic}: {len(hits)} results")
        return (angle.topic, [
            {"url": r["url"], "title": r.get("title", ""), "content": r.get("content", "")}
            for r in hits
        ])

    search_results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(search_one, a) for a in angles]
        for f in as_completed(futures):
            topic, hits = f.result()
            search_results[topic] = hits

    print(f"[batch_search] All searches complete")
    return {"search_results": search_results}


# ── Analyze Worker (Gemini only, reads pre-fetched search) ──

def analyze_worker(state: dict) -> dict:
    """
    Analyze pre-fetched search results with Gemini.
    No Tavily call here — search was done in batch_search.
    """
    company = state["company"]
    angle = state["angle"]
    search_hits = state.get("search_hits", [])

    if isinstance(angle, dict):
        angle = ResearchAngle(**angle)

    print(f"\n[analyze:{angle.topic}] Analyzing {len(search_hits)} sources...")

    if search_hits:
        context = "\n\n".join(
            f"[Source: {r['url']}]\nTitle: {r.get('title','')}\n{r.get('content','')}"
            for r in search_hits
        )
        sources = [r["url"] for r in search_hits]
        prompt = f"""You are a competitive intelligence researcher analyzing "{company}".
Research angle: {angle.topic}
Question: {angle.question}

Below are web search results. Analyze them to answer the research question.
Be specific: cite actual numbers, plan names, feature details.
Only include information supported by the search results.

SEARCH RESULTS:
{context}

Provide a detailed analysis in under 400 words."""
    else:
        sources = []
        prompt = f"""You are a competitive intelligence researcher analyzing "{company}".
Research angle: {angle.topic}
Question: {angle.question}
No web results available. Use training knowledge. Under 300 words."""

    response = llm.invoke(prompt)
    finding = ResearchFinding(topic=angle.topic, content=response.content, sources=sources)

    print(f"[analyze:{angle.topic}] Done ({len(sources)} sources)")
    bus.emit("worker_done", topic=angle.topic, sources_count=len(sources), is_redo=False, sources=sources)
    return {"findings": [finding]}


# ── Research Worker (full: search + analyze, used for redos) ──

def research_worker(state: dict) -> dict:
    company = state["company"]
    angle = state["angle"]
    feedback = state.get("feedback", "")
    if isinstance(angle, dict): angle = ResearchAngle(**angle)

    query = f"{company} {angle.question}"
    bus.emit("worker_start", topic=angle.topic, query=query, is_redo=True, feedback=feedback)
    print(f"\n[redo:{angle.topic}] Searching with refined query...")

    search_response = tavily_client.search(query=query, max_results=7, search_depth="advanced")
    results = search_response.get("results", [])

    if results:
        context = "\n\n".join(
            f"[Source: {r['url']}]\nTitle: {r.get('title','')}\n{r.get('content','')}"
            for r in results
        )
        sources = [r["url"] for r in results]
        prompt = f"""You are a competitive intelligence researcher analyzing "{company}".
Research angle: {angle.topic}
Question: {angle.question}

PREVIOUS ATTEMPT FLAGGED: {feedback}
Address this specifically.

SEARCH RESULTS:
{context}

Provide a detailed analysis in under 400 words."""
    else:
        sources = []
        prompt = f"""You are researching "{company}" — {angle.topic}: {angle.question}
No results available. Use training knowledge. Under 300 words."""

    response = llm.invoke(prompt)
    finding = ResearchFinding(topic=angle.topic, content=response.content, sources=sources)
    print(f"[redo:{angle.topic}] Done ({len(sources)} sources)")
    bus.emit("worker_done", topic=angle.topic, sources_count=len(sources), is_redo=True, sources=sources)
    return {"findings": [finding]}


# ── Critic ──────────────────────────────────────────

def critic(state: ScopeOutState) -> dict:
    findings = state["findings"]
    retry_count = state.get("retry_count", 0)

    bus.emit("critic_start", round=retry_count + 1)
    print(f"\n[critic] Evaluating {len(findings)} findings (round {retry_count+1})...")

    auto_flagged = {f.topic for f in findings if len(f.sources) < 2}

    findings_text = "\n\n---\n\n".join(
        f"Topic: {f.topic}\nWord count: {len(f.content.split())}\n"
        f"Sources: {len(f.sources)}\nPreview: {f.content[:500]}..."
        for f in findings
    )
    prompt = f"""You are a research quality critic. Evaluate each finding:
1. SOURCING: At least 2 distinct sources?
2. SPECIFICITY: Concrete numbers, names, prices?
3. DEPTH: 150+ words of real analysis?

FINDINGS:
{findings_text}

For each return: "topic", "verdict" (pass/redo), "reason" (if redo).
Return ONLY a JSON array."""

    response = llm.invoke(prompt)
    try:
        evaluations = _parse_json(response.content)
    except Exception:
        print("[critic] JSON parse failed, auto-passing")
        evaluations = [{"topic": f.topic, "verdict": "pass", "reason": ""} for f in findings]

    flagged_topics = set(auto_flagged)
    for ev in evaluations:
        if ev.get("verdict") == "redo": flagged_topics.add(ev["topic"])

    for ev in evaluations:
        topic = ev["topic"]
        is_flagged = topic in flagged_topics
        verdict = "redo" if is_flagged else "pass"
        reason = ev.get("reason", "")
        if is_flagged and not reason: reason = "insufficient sources"
        print(f"[critic] {topic}: {verdict.upper()}" + (f" — {reason}" if reason else ""))
        bus.emit("critic_verdict", topic=topic, verdict=verdict, reason=reason)

    if flagged_topics:
        print(f"[critic] {len(flagged_topics)} flagged")
    else:
        print("[critic] All passed!")
    bus.emit("critic_done", flagged=list(flagged_topics), round=retry_count + 1)
    return {"critique": evaluations, "flagged_topics": list(flagged_topics), "retry_count": retry_count + 1}


# ── Synthesizer (STREAMING) ─────────────────────────

def synthesizer(state: ScopeOutState) -> dict:
    """Stream report tokens to the browser in real time."""
    company = state["company"]
    findings = state["findings"]

    bus.emit("synthesizer_start")
    print(f"\n[synthesizer] Streaming report...")

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
- After each major claim, add a source reference as [source](url).
- Be direct and analytical.
- Every claim must be traceable to provided sources."""

    full_text = ""
    for chunk in llm.stream(prompt):
        if chunk.content:
            full_text += chunk.content
            bus.emit("report_chunk", text=chunk.content)

    print(f"[synthesizer] Done ({len(full_text)} chars)")
    bus.emit("synthesizer_done", report_length=len(full_text))
    return {"report": full_text}
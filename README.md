# ScopeOut — AI-Powered Competitive Intelligence Analyst

> Give it a company name. Watch AI agents research, critique, and write a professional competitive teardown — live.

ScopeOut is a multi-agent system built with **LangGraph** that produces sourced competitive teardown reports. Type a company name, and a team of specialized AI agents collaborates in real time: planning research angles, searching the web in parallel, critiquing quality, and synthesizing a polished report — all visible in a pixel-art office where you can watch them work.

[![Watch the demo](https://img.youtube.com/vi/yUvD6YIInpA/maxresdefault.jpg)](https://www.youtube.com/watch?v=yUvD6YIInpA)
---

## Architecture

```
                         ┌─────────────────────────────────────┐
                         │            ScopeOut Pipeline          │
                         └─────────────────────────────────────┘

  ┌──────────┐     ┌──────────────────────────┐     ┌──────────┐     ┌──────────────┐
  │  PLANNER │────▶│    PARALLEL WORKERS (×4)  │────▶│  CRITIC  │────▶│ SYNTHESIZER  │
  │  (Atlas) │     │  Pricing  │  Features     │     │  (Quinn) │     │   (Aria)     │
  │          │     │  Sentiment│  Positioning   │     │          │     │              │
  └──────────┘     └──────────────────────────┘     └────┬─────┘     └──────────────┘
                                                         │
                                                         │ redo flagged
                                                         ▼
                                                   ┌──────────┐
                                                   │  WORKER   │
                                                   │  (retry)  │
                                                   └──────────┘
```

### How It Works

1. **Planner** breaks the company into 4 research angles (pricing, features, sentiment, positioning)
2. **4 Parallel Workers** fan out simultaneously using LangGraph's `Send()` API — each searches the web via Tavily, then analyzes results with Gemini
3. **Critic** reviews every finding on sourcing quality, specificity, and depth. Weak findings get sent back to the relevant worker for a redo
4. **Synthesizer** combines all approved findings into a polished, source-cited competitive teardown

Every agent event streams to the browser via **Server-Sent Events (SSE)**, powering a real-time pixel-art office visualization.

---

## Key Engineering Patterns

| Pattern | Where | What It Demonstrates |
|---------|-------|---------------------|
| **Multi-agent collaboration** | Planner → Workers → Critic → Synthesizer | Specialized agents with distinct roles and handoffs |
| **Parallel fan-out** | `Send()` API dispatching 4 workers | LangGraph's concurrent execution with automatic result merging |
| **Critique-and-retry loop** | Critic → conditional edge → Worker redo | Self-improving quality gate with structured feedback |
| **Custom state reducer** | `_merge_findings()` | Upsert-by-topic so redone findings replace originals cleanly |
| **Real-time streaming** | FastAPI + SSE + EventBus | Thread-safe event queue bridging sync LangGraph nodes to async browser |
| **Tool integration** | Tavily web search | Grounding LLM analysis in current, sourced web data |

---

## Tech Stack

- **Orchestration:** LangGraph (state machines, `Send()` fan-out, conditional edges)
- **LLM:** Google Gemini 2.5 Flash (via `langchain-google-genai`)
- **Web Search:** Tavily (AI-optimized search API)
- **Backend:** FastAPI with Server-Sent Events
- **Frontend:** React 18 (CDN, no build step) with pixel-art office visualization
- **State:** Pydantic models + TypedDict with custom reducers

---

## Demo

### The Office — Agents at Work
Characters walk to the meeting room when researching, return to desks when done, have idle conversations, and visit the coffee room on breaks.

### The Report — Sourced Competitive Teardown
Every factual claim links to its web source. The critic ensures quality before the report is written.

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Google Gemini API key](https://aistudio.google.com/apikey) (free)
- [Tavily API key](https://tavily.com) (free)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/scopeout.git
cd scopeout
pip install -r requirements.txt
```

Create a `.env` file:
```
GOOGLE_API_KEY=your-gemini-key
TAVILY_API_KEY=your-tavily-key
```

### Run

```bash
# Web UI (recommended)
uvicorn server:app --reload
# Open http://localhost:8000

# CLI mode
python main.py "Spotify"
```

---

## Project Structure

```
scopeout/
├── state.py           # Pydantic models + TypedDict state with custom reducer
├── nodes.py           # Agent nodes: planner, worker, critic, synthesizer
├── graph.py           # LangGraph wiring with Send() fan-out + critic loop
├── event_bus.py       # Thread-safe event queue for SSE streaming
├── server.py          # FastAPI server with SSE endpoint
├── main.py            # CLI entry point
├── frontend/
│   └── index.html     # React pixel-art office dashboard (single file, no build)
├── requirements.txt
└── .env.example
```

---

## What I Learned Building This

This project was built as a level-up from a previous LangGraph pipeline (stateful ingredient analysis with conditional routing). ScopeOut adds three patterns that project didn't have:

1. **Parallel execution** — `Send()` with fan-out/fan-in is fundamentally different from sequential node chains. The `operator.add` reducer (later upgraded to `_merge_findings`) was the key design decision that made parallel workers possible without refactoring.

2. **Critique-and-retry** — A conditional edge that loops back isn't just an if-statement. Designing the state so redo findings *replace* originals (via the custom reducer) instead of duplicating them required rethinking the merge strategy.

3. **Real-time streaming** — Bridging synchronous LangGraph nodes (running in a thread pool) to an async FastAPI SSE endpoint through a thread-safe queue was the non-obvious architectural challenge behind the live visualization.

---

## License

MIT

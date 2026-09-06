"""
ScopeOut — Runner (Phase 4)
============================
Run from the project root:
    python main.py
    python main.py "Tesla Model 3"

Expects GOOGLE_API_KEY and TAVILY_API_KEY in your environment (or .env file).
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from graph import build_graph


def main():
    company = sys.argv[1] if len(sys.argv) > 1 else "Notion"

    print(f"\n{'=' * 60}")
    print(f"  ScopeOut — Competitive Teardown: {company}")
    print(f"{'=' * 60}")

    graph = build_graph()
    result = graph.invoke({
        "company": company,
        "retry_count": 0,
        "flagged_topics": [],
        "critique": [],
    })

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'-' * 60}")
    print(f"Research angles planned: {len(result['angles'])}")
    print(f"Findings collected:      {len(result['findings'])}")
    print(f"Quality review rounds:   {result.get('retry_count', 0)}")

    flagged = result.get("flagged_topics", [])
    if flagged:
        print(f"Topics sent for redo:    {', '.join(flagged)}")

    print(f"Report length:           {len(result['report'])} chars")
    print(f"{'-' * 60}\n")

    # ── Final report ──────────────────────────────────────
    print(result["report"])


if __name__ == "__main__":
    main()
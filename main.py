"""
ScopeOut — Runner (Phase 1)
============================
Run from the project root:
    python main.py

Expects GOOGLE_API_KEY in your environment (or in a .env file).
"""

import sys
from dotenv import load_dotenv
load_dotenv()
from graph import build_graph
def main(): # loads GOOGLE_API_KEY from .env if present

    # Default to "Notion" — swap in any company or product name
    company = sys.argv[1] if len(sys.argv) > 1 else "Notion"

    print(f"\n{'=' * 60}")
    print(f"  ScopeOut — Competitive Teardown: {company}")
    print(f"{'=' * 60}\n")

    graph = build_graph()
    result = graph.invoke({"company": company})

    # ── Summary of what happened ──────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Research angles planned: {len(result['angles'])}")
    print(f"Findings collected:      {len(result['findings'])}")
    print(f"Report length:           {len(result['report'])} chars")
    print(f"{'─' * 60}\n")

    # ── Final report ──────────────────────────────────────
    print(result["report"])


if __name__ == "__main__":
    main()
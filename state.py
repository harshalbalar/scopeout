"""
ScopeOut — State Schema (Phase 4)
==================================
Phase 4 upgrade: replaced operator.add with _merge_findings reducer
so redo workers overwrite weak findings instead of duplicating them.
Added critic loop state fields: critique, flagged_topics, retry_count.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


# ── Data Models (Pydantic) ──────────────────────────────────────────


class ResearchAngle(BaseModel):
    """A specific research angle the planner identified."""

    topic: str = Field(description="Short label, e.g. 'pricing', 'features'")
    question: str = Field(description="The concrete question to investigate")


class ResearchFinding(BaseModel):
    """Research output for one angle."""

    topic: str = Field(description="Which angle this finding covers")
    content: str = Field(description="The research content / analysis")
    sources: list[str] = Field(
        default_factory=list,
        description="Source URLs backing this finding",
    )


# ── Custom Reducer ──────────────────────────────────────────────────


def _merge_findings(
    existing: list[ResearchFinding], new: list[ResearchFinding]
) -> list[ResearchFinding]:
    """
    Merge findings by topic — newer entries overwrite older ones.

    This supports both:
      - Phase 3 parallel workers appending their own finding
      - Phase 4 redo workers replacing a weak finding with an improved one

    With operator.add, a redo would create a duplicate (old + new for
    the same topic). This reducer upserts by topic instead.
    """
    merged = {f.topic: f for f in existing}
    for f in new:
        merged[f.topic] = f
    return list(merged.values())


# ── Graph State (TypedDict) ─────────────────────────────────────────


class ScopeOutState(TypedDict):
    """
    The shared state that flows through the ScopeOut graph.

    Phase 4 additions:
      - critique:       structured evaluation from the critic
      - flagged_topics: topics that need redo (empty = all pass)
      - retry_count:    how many critic rounds have run (caps at 2)
    """

    company: str
    angles: list[ResearchAngle]
    findings: Annotated[list[ResearchFinding], _merge_findings]
    report: str

    # Phase 4: critic loop
    critique: list[dict]        # [{"topic": ..., "verdict": "pass"/"redo", "reason": ...}]
    flagged_topics: list[str]   # topics flagged for redo
    retry_count: int            # number of critic evaluations completed
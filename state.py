"""
ScopeOut — State Schema
========================
Defines the data models (Pydantic) and the graph state (TypedDict).

Design decision: TypedDict for the graph state so we can use LangGraph's
reducer annotations (e.g. operator.add to let parallel workers append
findings). Pydantic BaseModel for the data structures inside those channels,
giving us validation when the critic loop checks quality in Phase 4.
"""

from __future__ import annotations

import operator
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
        description="Source URLs — empty until Phase 2 adds web search",
    )


# ── Graph State (TypedDict) ─────────────────────────────────────────


class ScopeOutState(TypedDict):
    """
    The shared state that flows through the ScopeOut graph.

    - company:  the user's input (company or product name)
    - angles:   research angles produced by the planner
    - findings: research results — uses operator.add so parallel workers
                in Phase 3 can each append without overwriting each other
    - report:   the final synthesized teardown
    """

    company: str
    angles: list[ResearchAngle]
    findings: Annotated[list[ResearchFinding], operator.add]
    report: str

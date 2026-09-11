"""
ScopeOut — State Schema (Phase 5 optimized)
=============================================
Added search_results for batch search pre-fetch.
"""
from __future__ import annotations
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field


class ResearchAngle(BaseModel):
    topic: str = Field(description="Short label")
    question: str = Field(description="The concrete question to investigate")

class ResearchFinding(BaseModel):
    topic: str = Field(description="Which angle this covers")
    content: str = Field(description="The research content")
    sources: list[str] = Field(default_factory=list)

def _merge_findings(existing: list[ResearchFinding], new: list[ResearchFinding]) -> list[ResearchFinding]:
    merged = {f.topic: f for f in existing}
    for f in new:
        merged[f.topic] = f
    return list(merged.values())


class ScopeOutState(TypedDict):
    company: str
    angles: list[ResearchAngle]
    findings: Annotated[list[ResearchFinding], _merge_findings]
    report: str
    # Phase 4: critic loop
    critique: list[dict]
    flagged_topics: list[str]
    retry_count: int
    # Optimization: pre-fetched search results
    search_results: dict  # {topic: [{url, title, content}, ...]}
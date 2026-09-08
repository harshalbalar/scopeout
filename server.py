"""
ScopeOut — FastAPI Server (Phase 5a)
=====================================
Runs the LangGraph pipeline in a background thread and streams
real-time agent events to the frontend via Server-Sent Events.

Start with:
    uvicorn server:app --reload

Then open http://localhost:8000 in your browser.
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from event_bus import bus
from graph import build_graph

app = FastAPI(title="ScopeOut — Competitive Intelligence Analyst")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/analyze/{company}")
async def analyze(company: str):
    """
    Stream real-time agent events as the pipeline runs.

    The graph runs in a background thread (because LangGraph nodes
    are synchronous). Events emitted by nodes flow through the
    EventBus queue and get pushed to the browser as SSE.

    The stream ends when either a 'complete' or 'error' event fires.
    """
    bus.clear()

    graph = build_graph()

    def run_graph():
        try:
            result = graph.invoke({
                "company": company,
                "retry_count": 0,
                "flagged_topics": [],
                "critique": [],
            })
            bus.emit(
                "complete",
                report=result["report"],
                company=result["company"],
                findings_count=len(result["findings"]),
                retry_count=result.get("retry_count", 0),
                angles=[
                    a.model_dump() if hasattr(a, "model_dump") else a
                    for a in result.get("angles", [])
                ],
            )
        except Exception as e:
            bus.emit("error", message=str(e))

    thread = threading.Thread(target=run_graph, daemon=True)
    thread.start()

    async def event_stream():
        while True:
            # Poll the queue from an executor so we don't block the event loop
            event = await asyncio.get_event_loop().run_in_executor(
                None, lambda: bus.get(timeout=0.5)
            )
            if event:
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in ("complete", "error"):
                    break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML."""
    return FileResponse("frontend/index.html")

"""
ScopeOut — Event Bus
=====================
A thread-safe event queue that nodes emit to and the SSE endpoint
reads from. Uses queue.Queue so parallel workers (running in
separate threads via LangGraph) can all emit safely.

Single-request design: one active analysis at a time. Fine for a
portfolio project — production would use per-session channels.
"""

import queue
import json


class EventBus:
    def __init__(self):
        self._queue: queue.Queue = queue.Queue()

    def emit(self, event_type: str, **data):
        """Push an event from any thread (nodes call this)."""
        self._queue.put({"type": event_type, **data})

    def get(self, timeout: float = 0.5):
        """
        Blocking get with timeout. Returns None if no event
        arrives within the timeout — the SSE loop uses this
        to poll without busy-waiting.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self):
        """Drain any leftover events from a previous run."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


# Global instance — imported by nodes.py and server.py
bus = EventBus()

"""
Custom exceptions raised by the agent/retrieval layer.

Why these exist instead of just letting RateLimitError/ChromaError bubble up
raw: agent.py and retrieval.py shouldn't need to know anything about HTTP or
FastAPI. They raise a plain, app-specific exception; main.py (the only layer
that knows about HTTP) is what decides which status code that becomes.
This keeps your agent logic reusable outside of an API context too — e.g.
your Day 24 evaluate_agent script can import run_agent without dragging
in any FastAPI-specific error types.
"""


class QuotaExceededError(Exception):
    """Raised when the underlying LLM provider rate-limits us."""
    pass


class KnowledgeBaseError(Exception):
    """Raised when the vector store (Chroma) fails or is unreachable."""
    pass

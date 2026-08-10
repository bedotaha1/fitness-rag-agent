"""
Dependency provider functions.

Each of these is a thin pointer to something created ONCE at startup
(in main.py's lifespan) and stored on app.state — never a factory that
builds something new per request. Depends(get_x) is the seam: production
code reads from app.state, tests override the function itself via
app.dependency_overrides, with zero changes to route code either way.
"""
from fastapi import Request


def get_collection(request: Request):
    return request.app.state.collection


def get_seek_client(request: Request):
    return request.app.state.client_seek


def get_retrieve_fn(request: Request):
    """
    Returns your actual retrieve(query, collection) function from Day 18-20.
    Stored on app.state at startup too, for the same reason the collection
    and LLM client are — one seam, swappable in tests, created once.
    """
    return request.app.state.retrieve_fn


def get_client_id(request: Request):
    """Identifies a visitor by IP for the demo rate limit."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def get_request_counts(request: Request):
    """The in-memory per-IP request counter, stored on app.state."""
    return request.app.state.request_counts


def get_request(request: Request):
    """The raw request object, for reading cookies (pass-key unlock token)."""
    return request

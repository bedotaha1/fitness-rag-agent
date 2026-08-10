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

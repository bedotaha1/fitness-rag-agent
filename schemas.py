"""
Pydantic models: the declared shapes for every request and response body.

Same idea as your Day 12 structured outputs and your Day 22-23 tool schema —
declare the shape once, get validation + auto-docs for free at this boundary.
"""
from typing import Optional
from pydantic import BaseModel


class Question(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    context: str  # the retrieved chunks that were actually used, for transparency/debugging


class ErrorResponse(BaseModel):
    """
    One consistent error shape reused across every route.
    error_code is the machine-readable field a frontend actually branches on —
    message is human-readable, retry_after_seconds is only populated for 429s.
    """
    error_code: str
    message: str
    retry_after_seconds: Optional[int] = None

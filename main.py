"""
Main FastAPI app.

This is the ONLY module that knows about HTTP. agent.py and retrieval logic
raise plain Python exceptions (QuotaExceededError, KnowledgeBaseError);
this file is where those get translated into actual status codes — same
separation of concerns as "tool description states relevance, system prompt
states behavior" from Day 24: each module has exactly one job.
"""
import json
import secrets
from collections import defaultdict
from contextlib import asynccontextmanager
from os import getenv
from time import perf_counter

import chromadb
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI  # DeepSeek is OpenAI-Chat-Completions-compatible

load_dotenv()

from agent import judge_answer, run_agent, run_agent_stream
from dependencies import (
    get_client_id,
    get_collection,
    get_request,
    get_request_counts,
    get_retrieve_fn,
    get_seek_client,
)
from exceptions import KnowledgeBaseError, QuotaExceededError
from retrieval import retrieve  # <-- your actual Day 18-20 retrieve() function
from schemas import AnswerResponse, ErrorResponse, Question, UnlockRequest
from usage_log import get_stats, init_db, log_request

MAX_DEMO_REQUESTS = 3  # per visitor (by IP), for the life of the server process
UNLOCK_PASSKEY = getenv("UNLOCK_PASSKEY", "")  # read from .env; entering it in the frontend disables the demo limit
UNLOCK_COOKIE = "passkey_token"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- runs ONCE, at startup ----
    app.state.collection = chromadb.PersistentClient("chromaDB").get_or_create_collection(
        name="RAG_Pipeline", metadata={"hnsw:space": "cosine"}
    )
    app.state.client_seek = OpenAI(
        api_key=getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    app.state.retrieve_fn = retrieve
    app.state.request_counts = defaultdict(int)  # {client_ip: count}, in-memory, resets on restart
    app.state.unlocked_tokens: set[str] = set()  # pass-key tokens that bypass the demo limit
    init_db()

    yield
    # ---- runs ONCE, at shutdown ----
    # nothing to clean up yet — Chroma's PersistentClient doesn't need explicit closing,
    # but this is where you'd put it if a future dependency did


app = FastAPI(title="Fitness RAG Agent API", lifespan=lifespan)

# Allows a browser-based frontend (different origin, or a local file) to call
# this API. Wide open ("*") is fine for a portfolio/demo project — for a real
# production deployment you'd lock allow_origins down to your actual frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/ask",
    response_model=AnswerResponse,
    responses={
        429: {"model": ErrorResponse, "description": "Demo limit reached OR rate limited by the LLM provider"},
        503: {"model": ErrorResponse, "description": "Knowledge base temporarily unavailable"},
        500: {"model": ErrorResponse, "description": "Unexpected server error"},
    },
)
def ask(
    payload: Question,
    request: Request = Depends(get_request),
    collection=Depends(get_collection),
    client_seek=Depends(get_seek_client),
    retrieve_fn=Depends(get_retrieve_fn),
    client_id: str = Depends(get_client_id),
    request_counts: dict = Depends(get_request_counts),
):
    """
    Note: this is a plain `def`, not `async def` — deliberately (Option B from
    our sync/async discussion). run_agent's LLM calls are synchronous; FastAPI
    runs plain-def routes in its thread pool automatically, so a slow LLM call
    here doesn't freeze the whole server's event loop. If traffic ever demands
    true concurrency beyond the thread pool's size, this is the spot to revisit
    for a rewrite to AsyncOpenAI + async def all the way down.
    """
    token = request.cookies.get(UNLOCK_COOKIE)
    unlocked = bool(token and token in request.app.state.unlocked_tokens)

    # Demo rate limit is checked BEFORE anything that spends real quota —
    # a visitor who's already hit their limit never triggers run_agent at all.
    # Pass-key holders bypass this entirely.
    if not unlocked and request_counts[client_id] >= MAX_DEMO_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail={
                "error_code": "DEMO_LIMIT_REACHED",
                "message": f"This portfolio demo is limited to {MAX_DEMO_REQUESTS} questions per visitor "
                "to keep it available for everyone. Thanks for trying it out!",
            },
        )
    if not unlocked:
        request_counts[client_id] += 1
    remaining = None if unlocked else MAX_DEMO_REQUESTS - request_counts[client_id]

    start = perf_counter()

    try:
        answer, context = run_agent(client_seek, collection, retrieve_fn, payload.question)
        verdict = judge_answer(client_seek, payload.question, answer, context)
        latency_ms = round((perf_counter() - start) * 1000)

        log_request(
            question=payload.question,
            latency_ms=latency_ms,
            tool_called=bool(context.strip()),
            faithful=verdict["faithful"],
            status_code=200,
        )

        return AnswerResponse(
            answer=answer,
            context=context,
            requests_remaining=remaining,
            faithful=verdict["faithful"],
            faithfulness_reasoning=verdict["reasoning"],
        )

    except QuotaExceededError:
        log_request(payload.question, round((perf_counter() - start) * 1000), None, None, 429, "RATE_LIMITED")
        raise HTTPException(
            status_code=429,
            detail={
                "error_code": "RATE_LIMITED",
                "message": "The service is currently rate-limited. Please try again shortly.",
                "retry_after_seconds": 30,
            },
        )

    except KnowledgeBaseError:
        log_request(payload.question, round((perf_counter() - start) * 1000), None, None, 503, "KNOWLEDGE_BASE_UNAVAILABLE")
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "KNOWLEDGE_BASE_UNAVAILABLE",
                "message": "The knowledge base is temporarily unavailable.",
            },
        )

    except Exception:
        log_request(payload.question, round((perf_counter() - start) * 1000), None, None, 500, "INTERNAL_ERROR")
        # Catch-all MUST come last — Python matches except clauses top to
        # bottom, and a bare Exception listed first would swallow the two
        # more specific handlers above before they ever ran.
        raise HTTPException(
            status_code=500,
            detail={"error_code": "INTERNAL_ERROR", "message": "An unexpected error occurred."},
        )


@app.post("/ask/stream")
def ask_stream(
    payload: Question,
    request: Request = Depends(get_request),
    collection=Depends(get_collection),
    client_seek=Depends(get_seek_client),
    retrieve_fn=Depends(get_retrieve_fn),
    client_id: str = Depends(get_client_id),
    request_counts: dict = Depends(get_request_counts),
):
    """
    SSE variant of /ask — streams the answer token-by-token instead of
    waiting for the full response. Same rate-limit/unlock logic as /ask,
    but the body is a text/event-stream of events:
      event: meta   -> {requests_remaining}
      event: status -> {message}   (tool round in progress)
      (message)     -> {token}     (a piece of the answer, default event)
      event: done   -> {answer, context, faithful, ...}
      event: error  -> {error_code, message}
    """
    token = request.cookies.get(UNLOCK_COOKIE)
    unlocked = bool(token and token in request.app.state.unlocked_tokens)

    if not unlocked and request_counts[client_id] >= MAX_DEMO_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail={
                "error_code": "DEMO_LIMIT_REACHED",
                "message": f"This portfolio demo is limited to {MAX_DEMO_REQUESTS} questions per visitor "
                "to keep it available for everyone. Thanks for trying it out!",
            },
        )
    if not unlocked:
        request_counts[client_id] += 1
    remaining = None if unlocked else MAX_DEMO_REQUESTS - request_counts[client_id]

    def event_stream():
        start = perf_counter()
        yield f"event: meta\ndata: {json.dumps({'requests_remaining': remaining})}\n\n"

        try:
            for evt in run_agent_stream(client_seek, collection, retrieve_fn, payload.question):
                if evt["type"] == "status":
                    yield f"event: status\ndata: {json.dumps({'message': evt['message']})}\n\n"
                elif evt["type"] == "token":
                    yield f"data: {json.dumps({'token': evt['content']})}\n\n"
                elif evt["type"] == "done":
                    latency_ms = round((perf_counter() - start) * 1000)
                    answer = evt["answer"]
                    context = evt["context"]
                    verdict = judge_answer(client_seek, payload.question, answer, context)

                    log_request(
                        question=payload.question,
                        latency_ms=latency_ms,
                        tool_called=bool(context.strip()),
                        faithful=verdict["faithful"],
                        status_code=200,
                    )
                    yield (
                        "event: done\n"
                        "data: "
                        + json.dumps(
                            {
                                "answer": answer,
                                "context": context,
                                "requests_remaining": remaining,
                                "faithful": verdict["faithful"],
                                "faithfulness_reasoning": verdict["reasoning"],
                                "latency_ms": latency_ms,
                            }
                        )
                        + "\n\n"
                    )
        except QuotaExceededError:
            log_request(payload.question, round((perf_counter() - start) * 1000), None, None, 429, "RATE_LIMITED")
            yield f"event: error\ndata: {json.dumps({'error_code': 'RATE_LIMITED', 'message': 'The service is currently rate-limited. Please try again shortly.'})}\n\n"
        except KnowledgeBaseError:
            log_request(payload.question, round((perf_counter() - start) * 1000), None, None, 503, "KNOWLEDGE_BASE_UNAVAILABLE")
            yield f"event: error\ndata: {json.dumps({'error_code': 'KNOWLEDGE_BASE_UNAVAILABLE', 'message': 'The knowledge base is temporarily unavailable.'})}\n\n"
        except Exception:
            log_request(payload.question, round((perf_counter() - start) * 1000), None, None, 500, "INTERNAL_ERROR")
            yield f"event: error\ndata: {json.dumps({'error_code': 'INTERNAL_ERROR', 'message': 'An unexpected error occurred.'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/stats")
def stats():
    """
    Tiny analytics view — aggregate usage stats derived from usage_log.db.
    Not authenticated; fine for a portfolio demo, would need auth in a real
    production deployment before exposing usage data publicly.
    """
    return get_stats()


@app.get("/health")
def health():
    """Cheap liveness check — no LLM or Chroma calls, just confirms the process is up."""
    return {"status": "ok"}


@app.post("/unlock", responses={403: {"model": ErrorResponse, "description": "Wrong pass key"}})
def unlock(payload: UnlockRequest, request: Request = Depends(get_request)):
    """
    Pass-key gate for the demo rate limit. A correct key issues an httponly
    cookie whose random token is recorded on app.state; /ask then recognizes
    that token and skips the per-visitor request cap. Tokens are unguessable
    and in-memory (lost on restart, like the counter itself).
    """
    if payload.passkey != UNLOCK_PASSKEY:
        raise HTTPException(
            status_code=403,
            detail={"error_code": "INVALID_PASSKEY", "message": "That pass key isn't valid."},
        )
    token = secrets.token_hex(16)
    request.app.state.unlocked_tokens.add(token)
    response = JSONResponse(content={"unlocked": True, "requests_remaining": None})
    response.set_cookie(UNLOCK_COOKIE, token, httponly=True, samesite="lax")
    return response


# Serves static/index.html at "/" — the console frontend. Mounted last so it
# doesn't shadow the /ask and /health routes defined above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")

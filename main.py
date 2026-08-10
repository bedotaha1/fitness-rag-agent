"""
Main FastAPI app.

This is the ONLY module that knows about HTTP. agent.py and retrieval logic
raise plain Python exceptions (QuotaExceededError, KnowledgeBaseError);
this file is where those get translated into actual status codes — same
separation of concerns as "tool description states relevance, system prompt
states behavior" from Day 24: each module has exactly one job.
"""
from contextlib import asynccontextmanager
from os import getenv

import chromadb
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI  # DeepSeek is OpenAI-Chat-Completions-compatible

load_dotenv()

from agent import run_agent
from dependencies import get_collection, get_retrieve_fn, get_seek_client
from exceptions import KnowledgeBaseError, QuotaExceededError
from retrieval import retrieve  # <-- your actual Day 18-20 retrieve() function
from schemas import AnswerResponse, ErrorResponse, Question


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
        429: {"model": ErrorResponse, "description": "Rate limited by the LLM provider"},
        503: {"model": ErrorResponse, "description": "Knowledge base temporarily unavailable"},
        500: {"model": ErrorResponse, "description": "Unexpected server error"},
    },
)
def ask(
    payload: Question,
    collection=Depends(get_collection),
    client_seek=Depends(get_seek_client),
    retrieve_fn=Depends(get_retrieve_fn),
):
    """
    Note: this is a plain `def`, not `async def` — deliberately (Option B from
    our sync/async discussion). run_agent's LLM calls are synchronous; FastAPI
    runs plain-def routes in its thread pool automatically, so a slow LLM call
    here doesn't freeze the whole server's event loop. If traffic ever demands
    true concurrency beyond the thread pool's size, this is the spot to revisit
    for a rewrite to AsyncOpenAI + async def all the way down.
    """
    try:
        answer, context = run_agent(client_seek, collection, retrieve_fn, payload.question)
        return AnswerResponse(answer=answer, context=context)

    except QuotaExceededError:
        raise HTTPException(
            status_code=429,
            detail={
                "error_code": "RATE_LIMITED",
                "message": "The service is currently rate-limited. Please try again shortly.",
                "retry_after_seconds": 30,
            },
        )

    except KnowledgeBaseError:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "KNOWLEDGE_BASE_UNAVAILABLE",
                "message": "The knowledge base is temporarily unavailable.",
            },
        )

    except Exception:
        # Catch-all MUST come last — Python matches except clauses top to
        # bottom, and a bare Exception listed first would swallow the two
        # more specific handlers above before they ever ran.
        raise HTTPException(
            status_code=500,
            detail={"error_code": "INTERNAL_ERROR", "message": "An unexpected error occurred."},
        )


@app.get("/health")
def health():
    """Cheap liveness check — no LLM or Chroma calls, just confirms the process is up."""
    return {"status": "ok"}


# Serves static/index.html at "/" — the console frontend. Mounted last so it
# doesn't shadow the /ask and /health routes defined above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")

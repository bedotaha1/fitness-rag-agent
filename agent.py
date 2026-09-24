"""
The agent loop itself — this is your Day 22-24 code, structurally unchanged.
Only two things are different from your original, both deliberate:

1. call_seek no longer calls sys.exit() on a rate limit. In a script, killing
   the process was harmless. Inside a long-running API server, sys.exit()
   would take down every other in-flight request along with the one that
   tripped the limit — so it now raises QuotaExceededError instead, and lets
   the caller (main.py, the only layer that knows about HTTP) decide what
   that becomes for the client (a 429).

2. retrieve() failures are allowed to propagate as KnowledgeBaseError rather
   than an unhandled ChromaDB exception, for the same reason — this module
   shouldn't need to know it's being called from an API at all.
"""
from json import loads
from types import SimpleNamespace
from typing import Any

from exceptions import QuotaExceededError, KnowledgeBaseError

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the fitness knowledge base tool. "
    "The knowledge base covers exactly six topics: strength training, cardio, recovery, sleep, "
    "nutrition, and injury prevention. If the user's question is not about one of these six topics, "
    "do not call the tool, regardless of how confident or not confident you are in answering it "
    "yourself — respond that the question is outside the scope of this knowledge base. "
    "If the user's question is ambiguous within these topics (e.g. it could refer to more than one "
    "thing, like 'grams' without specifying protein, carbs, or fat), ask a single clarifying question "
    "first instead of searching broadly. "
    "Only state numeric guidance that appears in the retrieved context. Do not add comparison figures "
    "or facts (such as sedentary-person intake numbers) unless they were part of what was actually "
    "retrieved. "
    "If the tool returns no relevant information, say so honestly rather than guessing."
)

tools: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_fitness_knowledge",
            "description": (
                "Search the fitness knowledge base for information about strength training, "
                "cardio, recovery, sleep, nutrition, or injury prevention. Do not use for topics "
                "outside these six areas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to look up"}
                },
                "required": ["query"],
            },
        },
    }
]


def call_seek(client_seek: Any, history: list[Any], tools: list[Any] = []) -> Any:
    """
    Thin wrapper around the DeepSeek chat completion call.

    NOTE: client_seek is now a parameter instead of a module-level global —
    this is what makes it possible to inject a fake client in tests later,
    the same Depends()-based swap pattern from Day 25, just applied one
    layer down from the route itself.

    Import your actual RateLimitError type from wherever you get it from
    (openai.RateLimitError, since DeepSeek is OpenAI-Chat-Completions-compatible).
    """
    from openai import RateLimitError  # adjust import path if yours differs

    try:
        response = client_seek.chat.completions.create(
            model="deepseek-v4-flash",
            messages=history,
            tools=tools,
        )
        return response
    except RateLimitError:
        raise QuotaExceededError("DeepSeek API rate limit reached.")


def judge_answer(client_seek: Any, question: str, answer: str, context: str) -> dict:
    """
    LLM-as-judge faithfulness check (Day 20/24 concept), now run live per
    request instead of only in an offline eval script.

    Judges the answer against the retrieved CONTEXT, not against factual
    correctness in general — an answer can be "faithful" (grounded in what
    was retrieved) even if the retrieved context itself is wrong or thin.
    Honest "I don't know" / "outside scope" responses count as faithful,
    since they're not making anything up.

    Skipped entirely when context is empty (out-of-scope or conversational
    questions) — there's nothing to judge groundedness against, so the judge
    isn't called and no extra quota is spent on those turns.
    """
    if not context.strip():
        return {"faithful": None, "reasoning": "No retrieval occurred — nothing to judge against."}

    judge_prompt = (
        "You are a strict faithfulness judge. Given a QUESTION, the ANSWER a system gave, "
        "and the CONTEXT it was allowed to use, decide whether the ANSWER is fully grounded "
        "in the CONTEXT — no invented facts, no numbers or claims that don't appear in CONTEXT. "
        "An honest 'I don't know' or refusal counts as FAITHFUL. "
        "Respond with exactly one word first — FAITHFUL or UNFAITHFUL — then a dash and a "
        "one-sentence reason. Example: 'FAITHFUL - the protein range matches the context exactly.'"
    )

    history = [
        {"role": "system", "content": judge_prompt},
        {
            "role": "user",
            "content": f"QUESTION: {question}\n\nCONTEXT: {context}\n\nANSWER: {answer}",
        },
    ]

    response = call_seek(client_seek, history=history, tools=[])
    verdict_text = response.choices[0].message.content.strip()
    is_faithful = verdict_text.upper().startswith("FAITHFUL")

    return {"faithful": is_faithful, "reasoning": verdict_text}


def run_agent(
    client_seek: Any,
    collection: Any,
    retrieve_fn: Any,
    user_message: str,
    max_tool_calls: int = 4,
) -> tuple[str, str]:
    """
    Unchanged from your Day 22-24 version, except:
      - client_seek and retrieve_fn are passed in rather than read from module
        globals, so this function has no hidden dependencies on how the
        caller wired things up (testable, swappable — same DI principle).
      - Chroma failures inside retrieve_fn are caught and re-raised as
        KnowledgeBaseError, so main.py can turn them into a clean 503
        instead of an unhandled 500 with a raw ChromaDB traceback leaking
        to the client.
    """
    history: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    used_context: list[str] = []
    tool_call_count = 0

    while True:
        response = call_seek(client_seek, history=history, tools=tools)
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content, "\n\n".join(used_context)

        if tool_call_count >= max_tool_calls:
            history.append(
                {
                    "role": "user",
                    "content": "Please answer now using only the information already retrieved, "
                    "without calling any more tools.",
                }
            )
            final = call_seek(client_seek, history=history, tools=[])
            return final.choices[0].message.content, "\n\n".join(used_context)

        history.append(message)

        for call in message.tool_calls:
            if tool_call_count >= max_tool_calls:
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": "Tool call limit reached — not executed.",
                    }
                )
                continue
            tool_call_count += 1
            if call.function.name == "search_fitness_knowledge":
                args = loads(call.function.arguments)
                try:
                    chunks = retrieve_fn(args["query"], collection)
                except Exception as e:
                    raise KnowledgeBaseError(f"Retrieval failed: {e}") from e

                result_str = (
                    "\n\n".join(chunks) if chunks else "No relevant information found in the knowledge base."
                )
                used_context.append(result_str)

                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result_str,
                    }
                )


def _stream_chat(client_seek: Any, history: list[Any], tools: list[Any] = []) -> Any:
    """
    Streaming version of call_seek. Yields `{"type": "token", "content": ...}`
    events as the answer text arrives, then returns (full_content, tool_calls)
    once the stream ends. Tool-call arguments arrive split across deltas, so
    they're accumulated here into a complete call each.
    """
    from openai import RateLimitError  # adjust import path if yours differs

    try:
        stream = client_seek.chat.completions.create(
            model="deepseek-v4-flash",
            messages=history,
            tools=tools,
            stream=True,
        )
    except RateLimitError:
        raise QuotaExceededError("DeepSeek API rate limit reached.")

    content_parts: list[str] = []
    tool_deltas: dict[int, dict[str, str]] = {}

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if delta.content:
            content_parts.append(delta.content)
            yield {"type": "token", "content": delta.content}

        if delta.tool_calls:
            for tc in delta.tool_calls:
                bucket = tool_deltas.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    bucket["id"] += tc.id
                if tc.function:
                    if tc.function.name:
                        bucket["name"] += tc.function.name
                    if tc.function.arguments:
                        bucket["arguments"] += tc.function.arguments

    tool_calls = []
    for idx in sorted(tool_deltas):
        d = tool_deltas[idx]
        tool_calls.append(
            SimpleNamespace(
                id=d["id"],
                function=SimpleNamespace(name=d["name"], arguments=d["arguments"]),
            )
        )

    return "".join(content_parts), tool_calls


def _tool_call_dict(call: Any) -> dict:
    """Reconstruct the API-shaped assistant tool_call entry for history."""
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.function.name, "arguments": call.function.arguments},
    }


def run_agent_stream(
    client_seek: Any,
    collection: Any,
    retrieve_fn: Any,
    user_message: str,
    max_tool_calls: int = 4,
) -> Any:
    """
    Streaming twin of run_agent. A generator that yields event dicts:
      {"type": "status", "message": "..."}   -- tool round in progress
      {"type": "token",  "content": "..."}   -- a piece of the final answer
      {"type": "done", "answer": str, "context": str}
    Same tool-calling loop as run_agent, but the final answer is streamed
    token-by-token instead of arriving as one complete response.
    """
    history: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    used_context: list[str] = []
    tool_call_count = 0

    while True:
        content, tool_calls = yield from _stream_chat(client_seek, history, tools)

        if not tool_calls:
            yield {"type": "done", "answer": content, "context": "\n\n".join(used_context)}
            return

        if tool_call_count >= max_tool_calls:
            history.append(
                {
                    "role": "user",
                    "content": "Please answer now using only the information already retrieved, "
                    "without calling any more tools.",
                }
            )
            final_content, _ = yield from _stream_chat(client_seek, history, [])
            yield {"type": "done", "answer": final_content, "context": "\n\n".join(used_context)}
            return

        history.append(
            {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [_tool_call_dict(c) for c in tool_calls],
            }
        )

        for call in tool_calls:
            if tool_call_count >= max_tool_calls:
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": "Tool call limit reached — not executed.",
                    }
                )
                continue
            tool_call_count += 1
            if call.function.name == "search_fitness_knowledge":
                args = loads(call.function.arguments)
                try:
                    chunks = retrieve_fn(args["query"], collection)
                except Exception as e:
                    raise KnowledgeBaseError(f"Retrieval failed: {e}") from e

                result_str = (
                    "\n\n".join(chunks) if chunks else "No relevant information found in the knowledge base."
                )
                used_context.append(result_str)

                yield {"type": "status", "message": f"Searching knowledge base for: {args['query']}"}

                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result_str,
                    }
                )

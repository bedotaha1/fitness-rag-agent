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

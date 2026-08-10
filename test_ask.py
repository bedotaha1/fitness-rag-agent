"""
Example test using dependency_overrides — the exact seam we discussed.

Run with: pytest test_ask.py

Notice none of these tests touch a real Chroma database or spend real
DeepSeek quota. That's the entire payoff of Depends() from Day 25.
"""
from collections import defaultdict

from fastapi.testclient import TestClient

from dependencies import get_client_id, get_collection, get_request_counts, get_retrieve_fn, get_seek_client
from main import app
from usage_log import init_db

# lifespan (where init_db() normally runs) doesn't fire under a plain
# TestClient(app) — only under `with TestClient(app) as client:`. Calling it
# directly here is safe and idempotent (CREATE TABLE IF NOT EXISTS).
init_db()


class FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(FakeMessage(content))]


class FakeSeekClient:
    """Stands in for the DeepSeek/OpenAI client — no network call, no quota spent."""

    class chat:
        class completions:
            @staticmethod
            def create(model, messages, tools):
                # Always answers directly, never calls a tool — good enough
                # to test the /ask route's plumbing in isolation.
                return FakeResponse("This is a fake answer for testing.")


def fake_retrieve(query, collection):
    return ["Fake retrieved chunk about protein intake."]


# --- Wire the overrides before any test runs ---
app.dependency_overrides[get_collection] = lambda: None  # unused by the fake client path
app.dependency_overrides[get_seek_client] = lambda: FakeSeekClient()
app.dependency_overrides[get_retrieve_fn] = lambda: fake_retrieve
app.dependency_overrides[get_client_id] = lambda: "test-client-ip"

# A fresh dict per test run — real request_counts on app.state only exists
# once lifespan has run, which plain TestClient(app) doesn't trigger.
_test_request_counts = defaultdict(int)
app.dependency_overrides[get_request_counts] = lambda: _test_request_counts

client = TestClient(app)


def test_ask_returns_200_and_answer():
    response = client.post("/ask", json={"question": "How much protein do I need?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert body["answer"] == "This is a fake answer for testing."


def test_ask_rejects_malformed_body():
    # Missing the required "question" field entirely — Pydantic should 422 this
    # before the route body ever runs.
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_demo_limit_blocks_after_max_requests():
    # Fresh counter, isolated to this test, so it doesn't interfere with
    # the tests above that also hit /ask under the same overridden client_id.
    app.dependency_overrides[get_request_counts] = lambda: defaultdict(int)
    fresh_counts = app.dependency_overrides[get_request_counts]()
    app.dependency_overrides[get_request_counts] = lambda: fresh_counts

    for i in range(3):
        response = client.post("/ask", json={"question": f"question {i}"})
        assert response.status_code == 200
        assert response.json()["requests_remaining"] == 2 - i

    # 4th request from the same client_id should be blocked, no LLM call made
    response = client.post("/ask", json={"question": "one too many"})
    assert response.status_code == 429
    assert response.json()["detail"]["error_code"] == "DEMO_LIMIT_REACHED"

    # restore the shared counter for any tests that might run after this one
    app.dependency_overrides[get_request_counts] = lambda: _test_request_counts

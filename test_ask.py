"""
Example test using dependency_overrides — the exact seam we discussed.

Run with: pytest test_ask.py

Notice none of these tests touch a real Chroma database or spend real
DeepSeek quota. That's the entire payoff of Depends() from Day 25.
"""
from fastapi.testclient import TestClient

from dependencies import get_collection, get_retrieve_fn, get_seek_client
from main import app


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

"""
Logs every /ask call to a small local SQLite database — question, latency,
whether the tool was called, faithfulness verdict, status outcome.

Kept deliberately simple: SQLite needs no separate server, and this project's
traffic (a portfolio demo, rate-limited to 3 requests/visitor) will never
come close to needing anything heavier. Same "pick the right tool for the
actual scale" reasoning as choosing SQLite over Postgres for something this
small — no point running infrastructure you don't need yet.
"""
import sqlite3
import time
from contextlib import contextmanager

DB_PATH = "usage_log.db"


def init_db():
    """Called once at startup (from main.py's lifespan) — creates the table if it doesn't exist yet."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            question TEXT NOT NULL,
            latency_ms INTEGER,
            tool_called INTEGER,       -- 0/1, SQLite has no native boolean
            faithful INTEGER,          -- 0/1, NULL if no retrieval occurred
            status_code INTEGER NOT NULL,
            error_code TEXT            -- NULL on success
        )
        """
    )
    conn.commit()
    conn.close()


@contextmanager
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def log_request(
    question: str,
    latency_ms: int,
    tool_called: bool | None,
    faithful: bool | None,
    status_code: int,
    error_code: str | None = None,
):
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO requests (timestamp, question, latency_ms, tool_called, faithful, status_code, error_code)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                question,
                latency_ms,
                None if tool_called is None else int(tool_called),
                None if faithful is None else int(faithful),
                status_code,
                error_code,
            ),
        )
        conn.commit()


def get_stats() -> dict:
    """Aggregate stats for the /stats endpoint — the tiny analytics view."""
    with _get_conn() as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) as c FROM requests").fetchone()["c"]

        if total == 0:
            return {
                "total_requests": 0,
                "avg_latency_ms": None,
                "tool_call_rate": None,
                "faithfulness_rate": None,
                "error_rate": None,
                "top_rejected_questions": [],
            }

        avg_latency = conn.execute("SELECT AVG(latency_ms) as a FROM requests WHERE status_code = 200").fetchone()["a"]

        tool_calls = conn.execute("SELECT COUNT(*) as c FROM requests WHERE tool_called = 1").fetchone()["c"]
        judged_total = conn.execute("SELECT COUNT(*) as c FROM requests WHERE tool_called IS NOT NULL").fetchone()["c"]

        faithful_count = conn.execute("SELECT COUNT(*) as c FROM requests WHERE faithful = 1").fetchone()["c"]
        judged_faithfulness = conn.execute("SELECT COUNT(*) as c FROM requests WHERE faithful IS NOT NULL").fetchone()["c"]

        errors = conn.execute("SELECT COUNT(*) as c FROM requests WHERE status_code != 200").fetchone()["c"]

        # Questions that got NO retrieval (tool_called = 0) — a rough proxy
        # for "things people asked that the knowledge base doesn't cover."
        rejected = conn.execute(
            "SELECT question, COUNT(*) as c FROM requests WHERE tool_called = 0 "
            "GROUP BY question ORDER BY c DESC LIMIT 5"
        ).fetchall()

        return {
            "total_requests": total,
            "avg_latency_ms": round(avg_latency, 0) if avg_latency is not None else None,
            "tool_call_rate": round(tool_calls / judged_total, 3) if judged_total else None,
            "faithfulness_rate": round(faithful_count / judged_faithfulness, 3) if judged_faithfulness else None,
            "error_rate": round(errors / total, 3),
            "top_rejected_questions": [{"question": r["question"], "count": r["c"]} for r in rejected],
        }

"""
Your actual Day 18-20 retrieve() function.

One change from your original: the `collection` default argument
(= RAGCollection) was dropped. That default referenced a module-level
global that only existed because your script created the collection at
import time. In this project, the collection is created once in main.py's
lifespan and handed in via Depends -- collection is always passed explicitly
now, so a default here would either crash on import (name doesn't exist) or
risk becoming the exact mutable/stale-default bug you flagged earlier in
the roadmap. Everything else -- including the single-nearest-neighbor
threshold check -- is unchanged from what you had.
"""
from typing import Any

import chromadb


def retrieve(question: str, collection: chromadb.Collection, n_results: int = 3) -> list[Any] | None:
    res = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    if res['distances'][0][0] > 0.6:
        return None
    return res["documents"][0]

"""
Isolates the latency question: is ~2s the raw DeepSeek API call itself,
or is something in the FastAPI/CORS/browser stack adding to it?

This bypasses FastAPI, CORS, the static file mount, and the browser's
fetch() entirely -- just times the bare LLM call.

Run from your fitness_api/ folder (venv activated):
    python time_test.py
"""
import time
from os import getenv

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

start = time.perf_counter()
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
elapsed = time.perf_counter() - start

print(f"Raw DeepSeek call took: {elapsed:.2f}s")
print(response.choices[0].message.content)
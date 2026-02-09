"""Grok model family configuration (xAI, OpenAI-compatible API)."""

import os
from openai import AsyncOpenAI, APIStatusError

CONFIG = {
    "name": "grok",
    "output_file": "grok_results.csv",
    "models": [
        "grok-4-1-fast-reasoning",
        "grok-4-1-fast-non-reasoning",
        "grok-4-fast-reasoning",
        "grok-4-fast-non-reasoning",
    ],
    "variants": {
        "Low_Creativity": {"temp": 0.4, "top_p": 1.0},
        "Standard_Real":  {"temp": 0.7, "top_p": 0.9},
        "High_Risk":      {"temp": 1.0, "top_p": 0.95},
        "Chaos":          {"temp": 1.3, "top_p": 1.0},
    },
    "semaphore_limit": 40,
    "num_passes": 2,
    "retries": 3,
    "buffer_size": 20,
}


def create_client():
    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        raise ValueError("Set XAI_API_KEY environment variable")
    return AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1")


async def call_api(client, model, prompt, settings):
    """Make an xAI chat completion. Returns (response_text, finish_reason)."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings["temp"],
        top_p=settings["top_p"],
        max_completion_tokens=2048,
    )
    text = resp.choices[0].message.content or "[EMPTY_RESPONSE]"
    finish_reason = resp.choices[0].finish_reason or "stop"
    return text, finish_reason

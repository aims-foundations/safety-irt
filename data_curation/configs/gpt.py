"""GPT model family configuration."""

import os
from openai import AsyncOpenAI, APIStatusError

CONFIG = {
    "name": "gpt",
    "output_file": "gpt_results.csv",
    "models": [
        "gpt-4.1-mini",
        "gpt-4o-mini",
        "gpt-4.1-nano",
    ],
    "variants": {
        "Low_Creativity": {"temp": 0.4, "top_p": 1.0},
        "Standard_Real":  {"temp": 0.7, "top_p": 0.9},
        "High_Risk":      {"temp": 1.0, "top_p": 0.95},
        "Chaos":          {"temp": 1.3, "top_p": 1.0},
    },
    "semaphore_limit": 50,
    "num_passes": 2,
    "retries": 3,
    "buffer_size": 50,
}


def create_client():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable")
    return AsyncOpenAI(api_key=api_key)


async def call_api(client, model, prompt, settings):
    """Make an OpenAI chat completion. Returns (response_text, finish_reason)."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings["temp"],
        top_p=settings["top_p"],
        max_completion_tokens=1024,
    )
    text = resp.choices[0].message.content or "[EMPTY_RESPONSE]"
    finish_reason = resp.choices[0].finish_reason
    return text, finish_reason

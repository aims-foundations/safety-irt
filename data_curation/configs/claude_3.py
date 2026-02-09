"""Claude 3 Haiku model configuration."""

import os
from anthropic import AsyncAnthropic, APIStatusError

CONFIG = {
    "name": "claude_3",
    "output_file": "claude_3_results.csv",
    "models": [
        "claude-3-haiku-20240307",
    ],
    "variants": {
        "Low_Creativity": {"temp": 0.4, "top_p": 1.0},
        "Standard_Real":  {"temp": 0.7, "top_p": 0.9},
        "High_Risk":      {"temp": 1.0, "top_p": 0.95},
    },
    "semaphore_limit": 3,
    "num_passes": 2,
    "retries": 5,
    "buffer_size": 20,
}


def create_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")
    return AsyncAnthropic(api_key=api_key)


async def call_api(client, model, prompt, settings):
    """Make an Anthropic messages call. Returns (response_text, finish_reason)."""
    resp = await client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings["temp"],
        top_p=settings["top_p"],
        max_tokens=1024,
    )
    text = resp.content[0].text if resp.content else "[EMPTY_RESPONSE]"
    finish_reason = resp.stop_reason or "stop_sequence"
    return text, finish_reason

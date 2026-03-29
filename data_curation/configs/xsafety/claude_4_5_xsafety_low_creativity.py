"""Claude 4.5 Haiku - Low Creativity variant for XSafety prompts (temperature only, no top_p)."""

import os
from anthropic import AsyncAnthropic, APIStatusError

CONFIG = {
    "name": "claude_4_5_xsafety_low_creativity",
    "output_file": "claude_4_5_xsafety_low_creativity_results.csv",
    "models": [
        "claude-haiku-4-5-20251001",
    ],
    "variants": {
        "Low_Creativity": {"temp": 0.4},
    },
    "semaphore_limit": 3,
    "num_passes": 1,
    "retries": 5,
    "buffer_size": 20,
}


def create_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")
    return AsyncAnthropic(api_key=api_key)


async def call_api(client, model, prompt, settings):
    """Make an Anthropic messages call (temperature only, no top_p).

    Returns (response_text, finish_reason).
    """
    resp = await client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings["temp"],
        max_tokens=1024,
    )
    text = resp.content[0].text if resp.content else "[EMPTY_RESPONSE]"
    finish_reason = resp.stop_reason or "stop_sequence"
    return text, finish_reason

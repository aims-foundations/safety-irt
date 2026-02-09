"""Claude 4.5 Haiku - High Risk variant only (top_p only, no temperature)."""

import os
from anthropic import AsyncAnthropic, APIStatusError

CONFIG = {
    "name": "claude_4_5_high_risk",
    "output_file": "claude_4.5_high_risk_results.csv",
    "models": [
        "claude-haiku-4-5-20251001",
    ],
    "variants": {
        "High_Risk": {"top_p": 0.95},
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
    """Make an Anthropic messages call (top_p only, no temperature).

    Returns (response_text, finish_reason).
    """
    resp = await client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        top_p=settings["top_p"],
        max_tokens=1024,
    )
    text = resp.content[0].text if resp.content else "[EMPTY_RESPONSE]"
    finish_reason = resp.stop_reason or "stop_sequence"
    return text, finish_reason

"""Gemini model family configuration (Google GenAI SDK)."""

import os
from google import genai
from google.genai import types

CONFIG = {
    "name": "gemini",
    "output_file": "gemini_results.csv",
    "models": [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-lite-preview",
        "gemini-2.0-flash",
    ],
    "variants": {
        "Low_Creativity": {"temp": 0.4, "top_p": 1.0},
        "Standard_Real":  {"temp": 0.7, "top_p": 0.9},
        "High_Risk":      {"temp": 1.0, "top_p": 0.95},
        "Chaos":          {"temp": 1.3, "top_p": 1.0},
    },
    "semaphore_limit": 40,
    # Gemini scripts did not use multi-pass in the original code
    "num_passes": 1,
    "retries": 3,
    "buffer_size": 50,
}


def create_client():
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY environment variable")
    return genai.Client(api_key=api_key)


async def call_api(client, model, prompt, settings):
    """Make a Gemini generate_content call. Returns (response_text, finish_reason)."""
    safety_config = [
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    ]

    gen_config = types.GenerateContentConfig(
        temperature=settings["temp"],
        top_p=settings["top_p"],
        max_output_tokens=1024,
        safety_settings=safety_config,
    )

    resp = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=gen_config,
    )

    text = resp.text if resp.text else "[BLOCKED_BY_API]"
    return text, "stop"

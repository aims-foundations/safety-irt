"""Qwen 2.5 model configuration (local vLLM server, OpenAI-compatible API).

Start the vLLM server first:
    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct --port 8234 --gpu-memory-utilization 0.9

Then run:
    python -m data_curation.test_takers.run --config qwen
"""

import os
from openai import AsyncOpenAI

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8234/v1")

CONFIG = {
    "name": "qwen",
    "output_file": "qwen_results.csv",
    "models": [
        "Qwen/Qwen2.5-7B-Instruct",
    ],
    "variants": {
        "Low_Creativity": {"temp": 0.4, "top_p": 1.0},
        "Standard_Real":  {"temp": 0.7, "top_p": 0.9},
        "High_Risk":      {"temp": 1.0, "top_p": 0.95},
        "Chaos":          {"temp": 1.3, "top_p": 1.0},
    },
    "semaphore_limit": 32,
    "num_passes": 2,
    "retries": 3,
    "buffer_size": 50,
}


def create_client():
    return AsyncOpenAI(api_key="not-needed", base_url=VLLM_BASE_URL)


async def call_api(client, model, prompt, settings):
    """Make a vLLM chat completion (OpenAI-compatible). Returns (response_text, finish_reason)."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings["temp"],
        top_p=settings["top_p"],
        max_tokens=1024,
    )
    text = resp.choices[0].message.content or "[EMPTY_RESPONSE]"
    finish_reason = resp.choices[0].finish_reason
    return text, finish_reason

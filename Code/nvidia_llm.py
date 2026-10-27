"""Shared NVIDIA NIM client helpers for the reproduction pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODELS = [
    "openai/gpt-oss-20b",
    "google/gemma-4-31b-it",
    "nvidia/nemotron-3-super-120b-a12b",
]


def make_client(env_file=None, timeout=120.0):
    if env_file:
        path = Path(env_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Environment file not found: {path}")
        load_dotenv(path)
    else:
        load_dotenv()

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set")
    return OpenAI(base_url=BASE_URL, api_key=api_key, timeout=timeout, max_retries=0)


def complete(client, model, prompt, temperature=1.0, top_p=1.0, max_tokens=512):
    request = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False,
    )
    # GPT-OSS reasoning tokens share the output budget. Low effort is the closest
    # practical match to the paper's non-reasoning generation setup.
    if model.startswith("openai/gpt-oss-"):
        request["reasoning_effort"] = "low"
    response = client.chat.completions.create(**request)
    return response.choices[0].message.content or ""

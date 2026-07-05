"""Minimal httpx client for a local Ollama server."""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv

DEFAULT_MODEL = "llama3.1:8b"


def get_ollama_host() -> str:
    load_dotenv()
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 300.0,
    json_mode: bool = False,
) -> str:
    payload: dict = {"model": model, "prompt": prompt, "stream": False}
    if json_mode:
        payload["format"] = "json"
    response = httpx.post(f"{get_ollama_host()}/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["response"]

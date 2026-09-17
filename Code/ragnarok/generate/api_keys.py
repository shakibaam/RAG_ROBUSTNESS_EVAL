"""Load API keys from environment / repo `.env` (never hardcode secrets)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_env() -> None:
    # Prefer repo-root .env; also accept legacy .env.local names.
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(REPO_ROOT / ".env.local")
    load_dotenv(".env.local")


def get_openai_api_key() -> str | None:
    _load_env()
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPEN_AI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )


def get_anthropic_api_key() -> str | None:
    _load_env()
    return os.getenv("ANTHROPIC_API_KEY")

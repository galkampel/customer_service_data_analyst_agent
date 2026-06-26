"""LLM configuration helpers for Nebius Token Factory.

This module centralizes model and connection resolution so agent components
can request models by role without hardcoding model IDs.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Load environment variables from a local .env file if present.
load_dotenv()


def _require_api_key() -> str:
    """Return the required Nebius API key or raise a clear error."""
    api_key = os.environ.get("NEBIUS_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "NEBIUS_API_KEY environment variable is not set. "
            "Set it before running the agent."
        )
    return api_key


def resolve_model_for_role(role: str) -> str:
    """Resolve the model ID for a logical role.

    Resolution order:
    1) Role-specific env var (for known roles)
    2) NEBIUS_MODEL
    3) Built-in DEFAULT_MODEL
    """
    role_key_map = {
        "main": "NEBIUS_MAIN_MODEL",
        "router": "NEBIUS_ROUTER_MODEL",
        "profile": "NEBIUS_PROFILE_MODEL",
        "recommender": "NEBIUS_RECOMMENDER_MODEL",
    }
    role_key = role_key_map.get(role)
    if role_key:
        role_model = os.environ.get(role_key, "").strip()
        if role_model:
            return role_model

    fallback = os.environ.get("NEBIUS_MODEL", "").strip()
    return fallback or DEFAULT_MODEL


def make_llm(role: str, temperature: float = 0.0) -> ChatOpenAI:
    """Build a ChatOpenAI client configured for Nebius Token Factory."""
    base_url = (
        os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL).strip()
        or DEFAULT_BASE_URL
    )
    return ChatOpenAI(
        model=resolve_model_for_role(role),
        base_url=base_url,
        api_key=_require_api_key(),
        temperature=temperature,
    )

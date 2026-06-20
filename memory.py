"""Persistent memory helpers for Task 2.

This module provides:
- Episodic memory via LangGraph SQLite checkpoints.
- User profile memory via per-session JSON files.
"""

from __future__ import annotations

import json
import sqlite3
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llm_config import make_llm

_DEFAULT_MEMORY_DIR = Path("memory")
_DEFAULT_PROFILES_DIR = _DEFAULT_MEMORY_DIR / "profiles"
_DEFAULT_CHECKPOINT_DB = _DEFAULT_MEMORY_DIR / "checkpoints.db"

_PROFILE_EXTRACT_SYSTEM_PROMPT = """Extract ONLY durable user-profile facts.

Rules:
- Return strict JSON object only.
- Include only facts explicitly supported by the conversation.
- Keep values concise.
- Ignore transient one-off requests.
- If no durable facts are present, return {}.

Preferred keys:
- preferences: list[str]
- frequent_categories: list[str]
- frequent_intents: list[str]
- constraints: list[str]
- notes: list[str]
"""


def _ensure_parent(path: Path) -> None:
    """Create the parent directory for a path if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _sanitize_session_id(session_id: str) -> str:
    """Normalize session ID for safe file naming."""
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in session_id.strip()
    )
    return cleaned or "default"


def _profile_path(session_id: str, profiles_dir: Path) -> Path:
    """Return profile path for a session ID."""
    return profiles_dir / f"{_sanitize_session_id(session_id)}.json"


def get_checkpointer(
    db_path: str = str(_DEFAULT_CHECKPOINT_DB),
) -> Any:
    """Return a SQLite-backed LangGraph checkpointer."""
    sqlite_module = import_module("langgraph.checkpoint.sqlite")
    sqlite_saver_cls = getattr(sqlite_module, "SqliteSaver")

    db_file = Path(db_path)
    _ensure_parent(db_file)
    conn = sqlite3.connect(db_file, check_same_thread=False)
    return sqlite_saver_cls(conn)


def list_sessions(db_path: str = str(_DEFAULT_CHECKPOINT_DB)) -> list[str]:
    """List checkpointed thread IDs from SQLite storage."""
    db_file = Path(db_path)
    if not db_file.exists():
        return []

    try:
        conn = sqlite3.connect(db_file)
        cur = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        )
        rows = [str(row[0]) for row in cur.fetchall() if row and row[0]]
        conn.close()
        return rows
    except sqlite3.Error:
        return []


def load_user_profile(
    session_id: str,
    profiles_dir: str = str(_DEFAULT_PROFILES_DIR),
) -> dict[str, Any]:
    """Load profile JSON for a session, returning default schema if missing."""
    base_dir = Path(profiles_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    path = _profile_path(session_id, base_dir)

    default_profile: dict[str, Any] = {
        "preferences": [],
        "frequent_categories": [],
        "frequent_intents": [],
        "constraints": [],
        "notes": [],
    }

    if not path.exists():
        return default_profile

    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if not isinstance(loaded, dict):
                return default_profile
            merged = dict(default_profile)
            merged.update(loaded)
            return merged
    except (OSError, json.JSONDecodeError):
        return default_profile


def save_user_profile(
    session_id: str,
    profile: dict[str, Any],
    profiles_dir: str = str(_DEFAULT_PROFILES_DIR),
) -> None:
    """Persist profile JSON atomically to avoid partial writes."""
    base_dir = Path(profiles_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    path = _profile_path(session_id, base_dir)

    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(base_dir),
        delete=False,
    ) as temp:
        json.dump(profile, temp, indent=2, ensure_ascii=True)
        temp.flush()
        temp_path = Path(temp.name)

    temp_path.replace(path)


def _merge_profile(
    current_profile: dict[str, Any],
    new_facts: dict[str, Any],
) -> dict[str, Any]:
    """Conservatively merge extracted facts into an existing profile."""
    merged = dict(current_profile)

    for key, value in new_facts.items():
        if value in (None, "", [], {}):
            continue

        if isinstance(value, list):
            current = merged.get(key, [])
            if not isinstance(current, list):
                current = [str(current)] if current else []
            normalized = [
                str(v).strip()
                for v in current + value
                if str(v).strip()
            ]
            merged[key] = sorted(set(normalized))
        else:
            merged[key] = str(value).strip()

    return merged


def update_user_profile(
    profile: dict[str, Any],
    user_text: str,
    assistant_text: str,
) -> dict[str, Any]:
    """Extract new durable facts from latest exchange and merge them."""
    user_snippet = user_text.strip()
    assistant_snippet = assistant_text.strip()
    if not user_snippet and not assistant_snippet:
        return profile

    conversation_excerpt = (
        f"User: {user_snippet[:800]}\n"
        f"Assistant: {assistant_snippet[:800]}"
    )

    try:
        llm = make_llm(role="profile", temperature=0.0)
        response = llm.invoke(
            [
                SystemMessage(content=_PROFILE_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=conversation_excerpt),
            ]
        )
        raw = str(response.content).strip()
        new_facts = json.loads(raw) if raw else {}
        if not isinstance(new_facts, dict):
            return profile
        return _merge_profile(profile, new_facts)
    except (ValueError, TypeError, json.JSONDecodeError, RuntimeError):
        # Never fail a user query because profile extraction failed.
        return profile


def format_profile_for_prompt(profile: dict[str, Any]) -> str:
    """Format profile to a compact prompt-friendly bullet list."""
    if not profile:
        return "No stored user profile facts."

    lines: list[str] = ["Stored user profile facts:"]
    for key in (
        "preferences",
        "frequent_categories",
        "frequent_intents",
        "constraints",
        "notes",
    ):
        value = profile.get(key)
        if not value:
            continue
        if isinstance(value, list):
            lines.append(f"- {key}: {', '.join(str(v) for v in value)}")
        else:
            lines.append(f"- {key}: {value}")

    if len(lines) > 1:
        return "\n".join(lines)
    return "No stored user profile facts."

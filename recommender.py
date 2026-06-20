"""Query recommendation helpers for the customer-service analyst agent.

The recommender suggests a concrete next dataset query, optionally refines a
pending suggestion, and uses deterministic rule checks for confirmation or
declination. It never executes dataset tools directly.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from openai import OpenAIError

from llm_config import make_llm
from memory import format_profile_for_prompt

logger = logging.getLogger(__name__)

FALLBACK_SUGGESTION = (
    "What is the distribution of intents in the REFUND category?"
)

CATEGORIES = (
    "ACCOUNT",
    "CANCEL",
    "CONTACT",
    "DELIVERY",
    "FEEDBACK",
    "INVOICE",
    "ORDER",
    "PAYMENT",
    "REFUND",
    "SHIPPING",
    "SUBSCRIPTION",
)

CONFIRM_KEYWORDS = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "sure",
        "ok",
        "okay",
        "proceed",
        "execute",
        "confirm",
        "approved",
    }
)

DECLINE_KEYWORDS = frozenset(
    {
        "no",
        "nope",
        "nah",
        "cancel",
        "skip",
        "stop",
    }
)

CONFIRM_PHRASES = (
    "do it",
    "go ahead",
    "sounds good",
    "let's do it",
    "run it",
    "go for it",
    "do that",
)

DECLINE_PHRASES = (
    "never mind",
    "forget it",
    "not now",
    "don't do",
    "no thanks",
)

SUGGESTION_SYSTEM_PROMPT = """You are a query recommender for Customer Service
Data Analyst Agent.

The agent analyzes the Bitext Customer Service dataset. Valid categories are:
{categories}.

Based on the user profile and recent conversation, suggest exactly one concrete
next query the user could ask. The query must be directly about the dataset,
specific enough for the existing data-analysis tools, and naturally connected
to the user's context.

User profile:
{profile_context}

Recent conversation:
{history_context}

Return only the suggested query text. Do not add markdown, preamble, or
explanation.
"""

REFINE_SYSTEM_PROMPT = """You are a query recommender for a Customer Service
Data Analyst Agent.

The current pending query suggestion is:
{pending}

The user asked to refine it this way:
{refinement}

Return exactly one revised executable dataset query. Do not add markdown,
preamble, or explanation.
"""


def _word_tokens(text: str) -> frozenset[str]:
    """Return lowercase word tokens for safe confirmation matching."""
    return frozenset(re.findall(r"\b\w+\b", text.lower()))


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    """Return True when text contains one of the bounded phrases."""
    normalized = text.lower()
    return any(
        re.search(r"\b" + re.escape(phrase) + r"\b", normalized)
        for phrase in phrases
    )


def is_confirmation(text: str) -> bool:
    """Return True when text clearly confirms a pending recommendation."""
    tokens = _word_tokens(text)
    return bool(tokens & CONFIRM_KEYWORDS) or _contains_phrase(
        text,
        CONFIRM_PHRASES,
    )


def is_declination(text: str) -> bool:
    """Return True when text clearly declines a pending recommendation."""
    tokens = _word_tokens(text)
    return bool(tokens & DECLINE_KEYWORDS) or _contains_phrase(
        text,
        DECLINE_PHRASES,
    )


def _format_history(messages: list[BaseMessage], limit: int = 6) -> str:
    """Format recent chat history compactly for the recommender prompt."""
    recent_messages = messages[-limit:]
    if not recent_messages:
        return "No conversation history yet."

    lines: list[str] = []
    for message in recent_messages:
        role = type(message).__name__.replace("Message", "")
        content = str(message.content).replace("\n", " ").strip()
        lines.append(f"{role}: {content[:300]}")
    return "\n".join(lines)


def _clean_suggestion(text: str) -> str:
    """Normalize model output into one plain query string."""
    suggestion = text.strip().strip('"').strip("'").strip()
    if not suggestion:
        return FALLBACK_SUGGESTION
    return suggestion


def generate_suggestion(
    messages: list[BaseMessage],
    user_profile: dict[str, Any],
) -> str:
    """Generate one concrete follow-up query from history and profile."""
    try:
        llm = make_llm(role="recommender", temperature=0.3)
        response = llm.invoke(
            [
                SystemMessage(
                    content=SUGGESTION_SYSTEM_PROMPT.format(
                        categories=", ".join(CATEGORIES),
                        profile_context=format_profile_for_prompt(
                            user_profile
                        ),
                        history_context=_format_history(messages),
                    )
                ),
                HumanMessage(content="Suggest one follow-up query."),
            ]
        )
        return _clean_suggestion(str(response.content))
    except (EnvironmentError, OpenAIError, RuntimeError, ValueError) as exc:
        logger.warning("Recommendation generation failed: %s", exc)
        return FALLBACK_SUGGESTION


def refine_suggestion(pending: str, refinement: str) -> str:
    """Refine a pending query suggestion from user feedback."""
    pending = pending.strip() or FALLBACK_SUGGESTION
    refinement = refinement.strip()
    if not refinement:
        return pending

    try:
        llm = make_llm(role="recommender", temperature=0.3)
        response = llm.invoke(
            [
                SystemMessage(
                    content=REFINE_SYSTEM_PROMPT.format(
                        pending=pending,
                        refinement=refinement,
                    )
                ),
                HumanMessage(content="Revise the pending suggestion."),
            ]
        )
        return _clean_suggestion(str(response.content))
    except (EnvironmentError, OpenAIError, RuntimeError, ValueError) as exc:
        logger.warning("Recommendation refinement failed: %s", exc)
        return pending


def format_suggestion_response(suggestion: str) -> str:
    """Format a suggested query for the user without executing it."""
    suggestion = _clean_suggestion(suggestion)
    return (
        f"Suggested query:\n\"{suggestion}\"\n\n"
        "Would you like me to run this query? Reply yes to run it, "
        "no to cancel, or describe how to change it."
    )

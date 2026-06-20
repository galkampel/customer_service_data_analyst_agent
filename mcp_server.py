"""Task 3 FastMCP server for the customer service analyst project.

FastMCP is a lightweight Python framework for exposing functions as MCP
tools over transports such as streamable-http.  This server publishes a
stateless tool interface over the Bitext customer-service dataset so MCP
clients can query categories, intent distributions, examples, summaries,
and keyword matches.
"""

from __future__ import annotations

import textwrap
from typing import Optional

from fastmcp import FastMCP

from data_loader import load_dataset
from tools import (
    get_intent_distribution as get_intent_distribution_impl,
    list_categories as list_categories_impl,
    search_responses as search_responses_impl,
    summarize_category as summarize_category_impl,
)

mcp = FastMCP(
    name="customer-service-analyst",
    instructions=(
        "Use these tools to analyze the Bitext customer service dataset. "
        "All exposed tools are stateless and safe for concurrent clients."
    ),
)


def _show_examples_stateless(
    n: int = 5,
    category: Optional[str] = None,
    intent: Optional[str] = None,
) -> str:
    """Stateless show_examples implementation for MCP clients.

    Notes:
    - Does not rely on shared `_STATE` from `tools.py`.
    - Applies filters directly against full dataset each call.
    - Uses deterministic sampling for reproducible client results.
    """
    df = load_dataset()
    if category:
        df = df[df["category"] == category.upper().strip()]
    if intent:
        df = df[df["intent"] == intent.lower().strip()]

    if df.empty:
        return "No rows match the requested filters."

    size = min(max(1, n), 50)
    sample = df.sample(size, random_state=42)
    lines: list[str] = [
        (
            f"Showing {len(sample)} example(s) "
            f"(category={category or 'any'}, intent={intent or 'any'}):"
        )
    ]
    for idx, (_, row) in enumerate(sample.iterrows(), start=1):
        lines.append(
            f"[{idx}] Category={row['category']} | Intent={row['intent']}\n"
            f"    Customer: {row['instruction']}\n"
            f"    Agent: {textwrap.shorten(row['response'], width=220)}"
        )
    return "\n\n".join(lines)


@mcp.tool()
def list_categories() -> str:
    """List all dataset categories with row counts."""
    return list_categories_impl()


@mcp.tool()
def get_intent_distribution(category: Optional[str] = None) -> str:
    """Return intent distribution (optionally for one category)."""
    return get_intent_distribution_impl(category=category)


@mcp.tool()
def show_examples(
    n: int = 5,
    category: Optional[str] = None,
    intent: Optional[str] = None,
) -> str:
    """Return sample rows with optional category/intent filters."""
    return _show_examples_stateless(n=n, category=category, intent=intent)


@mcp.tool()
def count_rows_for_intent(intent: str) -> str:
    """Count rows for one intent label using a stateless direct lookup."""
    normalized = intent.lower().strip()
    if not normalized:
        return "Intent cannot be empty."

    df = load_dataset()
    count = int((df["intent"] == normalized).sum())
    if count == 0:
        sample = ", ".join(sorted(df["intent"].unique())[:25])
        return f"Intent '{normalized}' not found. Example intents: {sample}"

    pct = (count / len(df)) * 100
    return (
        f"Intent '{normalized}': {count:,} rows "
        f"({pct:.2f}% of {len(df):,} total)."
    )


@mcp.tool()
def summarize_category(category: str, n_examples: int = 3) -> str:
    """Summarize category with counts and representative examples."""
    return summarize_category_impl(category=category, n_examples=n_examples)


@mcp.tool()
def search_responses(keyword: str, n: int = 5) -> str:
    """Search customer/agent text with literal keyword matching."""
    return search_responses_impl(
        keyword=keyword,
        n=n,
        use_working_set=False,
    )


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8000

    print(f"Starting MCP server on http://{host}:{port}")
    print("Transport: streamable-http")
    print("Tools:")
    print("- list_categories")
    print("- get_intent_distribution")
    print("- show_examples")
    print("- count_rows_for_intent")
    print("- summarize_category")
    print("- search_responses")

    mcp.run(transport="streamable-http", host=host, port=port)

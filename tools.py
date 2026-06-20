"""Task 1 analysis tools with clear descriptions and Pydantic schemas.

These tools are designed for reliable model tool-selection:
- explicit names
- unambiguous descriptions
- typed input schemas
"""

from __future__ import annotations

import textwrap
from typing import Optional

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from data_loader import load_dataset

# Stateful working-set container for multi-step chains in one query.
# A dict wrapper avoids using the `global` statement while preserving behavior.
_STATE: dict[str, Optional[pd.DataFrame]] = {"filtered_df": None}


def _get_working_df() -> pd.DataFrame:
    """Return current filtered working set or full dataset."""
    filtered_df = _STATE["filtered_df"]
    return filtered_df if filtered_df is not None else load_dataset()


def reset_filter() -> None:
    """Reset shared working set back to full dataset."""
    _STATE["filtered_df"] = None


class EmptyInput(BaseModel):
    """Schema for tools with no input arguments."""


class CategoryInput(BaseModel):
    """Schema for tools that require a category."""

    category: str = Field(
        ...,
        description=(
            "Category name such as REFUND, SHIPPING, or ACCOUNT "
            "(case-insensitive)."
        ),
    )


class IntentInput(BaseModel):
    """Schema for tools that require an intent label."""

    intent: str = Field(
        ...,
        description=(
            "Intent label such as get_refund or cancel_order "
            "(case-insensitive)."
        ),
    )


class ShowExamplesInput(BaseModel):
    """Schema for retrieving sample examples."""

    n: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of examples to return.",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category filter.",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Optional intent filter.",
    )


class SearchInput(BaseModel):
    """Schema for keyword search."""

    keyword: str = Field(
        ...,
        description=(
            "Keyword or phrase to search in instruction and response text."
        ),
    )
    n: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Max number of matches to return.",
    )
    use_working_set: bool = Field(
        default=True,
        description=(
            "When true, search current filtered working set if one exists; "
            "otherwise search the full dataset."
        ),
    )


class SummarizeInput(BaseModel):
    """Schema for category summarization."""

    category: str = Field(..., description="Category to summarize.")
    n_examples: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Representative examples to include.",
    )


class IntentDistributionInput(BaseModel):
    """Schema for intent distribution."""

    category: Optional[str] = Field(
        default=None,
        description="Optional category restriction.",
    )


def list_categories() -> str:
    """List all categories with row counts."""
    counts = load_dataset()["category"].value_counts().sort_index()
    lines = [f"  - {cat}: {cnt:,} rows" for cat, cnt in counts.items()]
    return "Dataset categories:\n" + "\n".join(lines)


def list_intents(category: str) -> str:
    """List intents under one category with counts."""
    df = load_dataset()
    cat = category.upper().strip()
    subset = df[df["category"] == cat]
    if subset.empty:
        available = ", ".join(sorted(df["category"].unique()))
        return (
            f"Category '{cat}' not found. "
            f"Available categories: {available}"
        )

    counts = subset["intent"].value_counts().sort_index()
    lines = [f"  - {intent}: {cnt:,} rows" for intent, cnt in counts.items()]
    return f"Intents in '{cat}':\n" + "\n".join(lines)


def count_rows() -> str:
    """Count rows in current working set or full dataset."""
    current = _get_working_df()
    total = len(load_dataset())
    if _STATE["filtered_df"] is not None:
        return (
            "Current filtered working set: "
            f"{len(current):,} rows (out of {total:,} total)."
        )
    return f"Full dataset: {len(current):,} rows."


def filter_by_intent(intent: str) -> str:
    """Filter working set by intent label for downstream tools."""
    df = load_dataset()
    normalized = intent.lower().strip()
    subset = df[df["intent"] == normalized]
    if subset.empty:
        sample = ", ".join(sorted(df["intent"].unique())[:25])
        return f"Intent '{normalized}' not found. Example intents: {sample}"

    _STATE["filtered_df"] = subset.copy()
    category = subset["category"].iloc[0]
    return (
        f"Filter applied: intent='{normalized}' (category={category}). "
        f"Rows: {len(subset):,}."
    )


def filter_by_category(category: str) -> str:
    """Filter working set by category for downstream tools."""
    df = load_dataset()
    cat = category.upper().strip()
    subset = df[df["category"] == cat]
    if subset.empty:
        available = ", ".join(sorted(df["category"].unique()))
        return (
            f"Category '{cat}' not found. "
            f"Available categories: {available}"
        )

    _STATE["filtered_df"] = subset.copy()
    intents = ", ".join(sorted(subset["intent"].unique()))
    return (
        f"Filter applied: category='{cat}'. "
        f"Rows: {len(subset):,}. Intents: {intents}"
    )


def show_examples(
    n: int = 5,
    category: Optional[str] = None,
    intent: Optional[str] = None,
) -> str:
    """Show sample instruction/response examples from the working set."""
    df = _get_working_df()
    if category:
        df = df[df["category"] == category.upper().strip()]
    if intent:
        df = df[df["intent"] == intent.lower().strip()]

    if df.empty:
        return "No rows match the requested filters."

    sample = df.sample(min(max(1, n), 50), random_state=42)
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


def search_responses(
    keyword: str,
    n: int = 5,
    use_working_set: bool = True,
) -> str:
    """Search instruction and response text by keyword."""
    df = _get_working_df() if use_working_set else load_dataset()
    kw = keyword.lower().strip()
    if not kw:
        return "Keyword cannot be empty."

    mask = (
        # Use literal substring matching (regex disabled) for predictable,
        # lower-cost search behavior on natural language keywords.
        df["instruction"].str.lower().str.contains(kw, na=False, regex=False)
        | df["response"].str.lower().str.contains(kw, na=False, regex=False)
    )
    matches = df[mask].head(max(1, min(n, 50)))
    if matches.empty:
        return f"No rows found for keyword '{keyword}'."

    lines = [
        (
            f"Found {int(mask.sum()):,} matches for '{keyword}'. "
            f"Showing {len(matches)}:"
        )
    ]
    for idx, (_, row) in enumerate(matches.iterrows(), start=1):
        lines.append(
            f"[{idx}] Category={row['category']} | Intent={row['intent']}\n"
            f"    Customer: {row['instruction']}\n"
            f"    Agent: {textwrap.shorten(row['response'], width=220)}"
        )
    return "\n\n".join(lines)


def get_intent_distribution(category: Optional[str] = None) -> str:
    """Return intent distribution with counts and percentages."""
    df = load_dataset()
    title = "Intent distribution across full dataset"
    if category:
        cat = category.upper().strip()
        df = df[df["category"] == cat]
        if df.empty:
            return f"Category '{cat}' not found."
        title = f"Intent distribution for category '{cat}'"

    counts = df["intent"].value_counts()
    total = int(counts.sum())
    lines = [
        f"  - {intent}: {cnt:,} ({(cnt / total) * 100:.1f}%)"
        for intent, cnt in counts.items()
    ]
    return f"{title} ({total:,} rows):\n" + "\n".join(lines)


def summarize_category(category: str, n_examples: int = 3) -> str:
    """Produce a compact category summary with stats and examples."""
    df = load_dataset()
    cat = category.upper().strip()
    subset = df[df["category"] == cat]
    if subset.empty:
        available = ", ".join(sorted(df["category"].unique()))
        return (
            f"Category '{cat}' not found. "
            f"Available categories: {available}"
        )

    counts = subset["intent"].value_counts()
    breakdown = [
        f"  - {intent}: {cnt:,} ({(cnt / len(subset)) * 100:.1f}%)"
        for intent, cnt in counts.items()
    ]

    n = min(max(1, n_examples), 10)
    examples: list[str] = []
    # Select representative examples from the top N intents by count.
    for intent_name in counts.index[:n]:
        row = subset[subset["intent"] == intent_name].iloc[0]
        examples.append(
            f"[{intent_name}]\n"
            f"  Customer: {row['instruction']}\n"
            f"  Agent: {textwrap.shorten(row['response'], width=260)}"
        )

    return (
        f"Summary of category '{cat}'\n"
        f"Total rows: {len(subset):,}\n"
        f"Distinct intents: {subset['intent'].nunique()}\n\n"
        "Intent breakdown:\n"
        + "\n".join(breakdown)
        + "\n\n"
        + "Representative examples:\n"
        + "\n\n".join(examples)
    )


def build_tools() -> list[StructuredTool]:
    """Build all Task 1 tools as StructuredTool instances."""
    return [
        StructuredTool.from_function(
            func=list_categories,
            name="list_categories",
            description=(
                "List all high-level categories in the dataset with row "
                "counts. Use when the user asks what categories exist."
            ),
            args_schema=EmptyInput,
        ),
        StructuredTool.from_function(
            func=list_intents,
            name="list_intents",
            description=(
                "List intents inside one category with row counts. Use when "
                "the user asks for sub-topics or intents in a category."
            ),
            args_schema=CategoryInput,
        ),
        StructuredTool.from_function(
            func=count_rows,
            name="count_rows",
            description=(
                "Count rows in the current working set. If a filter was "
                "applied, this returns filtered count; otherwise full size."
            ),
            args_schema=EmptyInput,
        ),
        StructuredTool.from_function(
            func=filter_by_intent,
            name="filter_by_intent",
            description=(
                "Filter working set by intent for follow-up calls. Typical "
                "chain: filter_by_intent, then count_rows or show_examples."
            ),
            args_schema=IntentInput,
        ),
        StructuredTool.from_function(
            func=filter_by_category,
            name="filter_by_category",
            description=(
                "Filter working set by category for follow-up calls like "
                "count_rows, show_examples, or get_intent_distribution."
            ),
            args_schema=CategoryInput,
        ),
        StructuredTool.from_function(
            func=show_examples,
            name="show_examples",
            description=(
                "Show sample rows with customer instruction and agent "
                "response, with optional category/intent filters."
            ),
            args_schema=ShowExamplesInput,
        ),
        StructuredTool.from_function(
            func=search_responses,
            name="search_responses",
            description=(
                "Search customer and agent text for a literal keyword or "
                "phrase. By default this searches the current working set "
                "if filtered; set use_working_set=false to search full "
                "dataset. Use when user wording does not map to exact "
                "intent names."
            ),
            args_schema=SearchInput,
        ),
        StructuredTool.from_function(
            func=get_intent_distribution,
            name="get_intent_distribution",
            description=(
                "Return intent counts and percentages, optionally within one "
                "category. Use for distribution or breakdown questions."
            ),
            args_schema=IntentDistributionInput,
        ),
        StructuredTool.from_function(
            func=summarize_category,
            name="summarize_category",
            description=(
                "Produce category summary with intent breakdown and examples. "
                "Use for open-ended summarization questions."
            ),
            args_schema=SummarizeInput,
        ),
    ]

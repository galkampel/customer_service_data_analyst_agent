"""Data access utilities for the Bitext customer-service dataset.

Task 1 requires repeatable, data-driven answers. This module ensures
a stable, fast dataset source by combining local parquet caching with
in-process memoization.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
CACHE_PATH = Path(__file__).parent / "data" / "bitext_cache.parquet"


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    """Load and normalize the dataset.

    Load order:
    1) Local parquet cache if present.
    2) HuggingFace dataset download, then save parquet cache.

    Returns:
        Normalized dataframe containing instruction/category/intent/response.
    """
    if CACHE_PATH.exists():
        df = pd.read_parquet(CACHE_PATH)
        return _normalize(df)

    from datasets import load_dataset as hf_load

    hf_ds = hf_load(DATASET_NAME, split="train")
    df = hf_ds.to_pandas()
    df = _normalize(df)
    os.makedirs(CACHE_PATH.parent, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical fields to avoid case-sensitive mismatches."""
    out = df.copy()
    out["category"] = out["category"].astype(str).str.upper().str.strip()
    out["intent"] = out["intent"].astype(str).str.lower().str.strip()
    return out


def get_categories() -> list[str]:
    """Return all categories sorted alphabetically."""
    return sorted(load_dataset()["category"].unique().tolist())


def get_intents(category: Optional[str] = None) -> list[str]:
    """Return intents, optionally restricted to one category."""
    df = load_dataset()
    if category:
        df = df[df["category"] == category.upper().strip()]
    return sorted(df["intent"].unique().tolist())

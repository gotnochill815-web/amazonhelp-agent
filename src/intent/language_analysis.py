
from collections import Counter
from typing import Iterable

import pandas as pd
from langdetect import detect, LangDetectException


def detect_language(text: str) -> str:
    """Detect the dominant language of a text."""
    
    if pd.isna(text) or not str(text).strip():
        return "unknown"

    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"
    except Exception:
        return "unknown"


def add_language_column(
    df: pd.DataFrame,
    text_column: str = "clean_text",
) -> pd.DataFrame:
    """Return a copy with detected language labels."""

    result = df.copy()

    result["language"] = result[text_column].map(
        detect_language
    )

    return result


def language_distribution(
    df: pd.DataFrame,
    language_column: str = "language",
) -> pd.DataFrame:
    """Return language counts and percentages."""

    counts = (
        df[language_column]
        .value_counts(dropna=False)
        .rename_axis("language")
        .reset_index(name="messages")
    )

    counts["percentage"] = (
        counts["messages"] / len(df) * 100
    )

    return counts


def english_subset(
    df: pd.DataFrame,
    language_column: str = "language",
) -> pd.DataFrame:
    """Return only English messages."""

    return df[
        df[language_column].eq("en")
    ].copy()


from __future__ import annotations

import pandas as pd


def summarize_context_coverage(
    golden: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize how much conversation context is available."""

    if "context_available" not in golden.columns:
        raise ValueError(
            "Missing context_available column."
        )

    counts = (
        golden["context_available"]
        .fillna(False)
        .value_counts()
        .rename_axis("context_available")
        .reset_index(name="examples")
    )

    counts["percentage"] = (
        counts["examples"]
        / len(golden)
        * 100
    )

    return counts


def summarize_selection_reasons(
    golden: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize how examples entered the Golden Set."""

    return (
        golden["selection_reason"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("selection_reason")
        .reset_index(name="examples")
    )


def summarize_languages(
    golden: pd.DataFrame,
) -> pd.DataFrame:
    """
    Crude language grouping based on Unicode ranges.

    This is only for sanity checking, not language identification.
    """

    def detect_script(text: str) -> str:
        if pd.isna(text):
            return "unknown"

        text = str(text)

        if any(
            "\u3040" <= ch <= "\u30ff"
            for ch in text
        ):
            return "japanese"

        if any(
            "\u4e00" <= ch <= "\u9fff"
            for ch in text
        ):
            return "cjk"

        if any(
            "\u0400" <= ch <= "\u04ff"
            for ch in text
        ):
            return "cyrillic"

        return "latin"

    result = golden.copy()

    result["script_group"] = result[
        "customer_message"
    ].map(detect_script)

    counts = (
        result["script_group"]
        .value_counts()
        .rename_axis("script_group")
        .reset_index(name="examples")
    )

    counts["percentage"] = (
        counts["examples"]
        / len(result)
        * 100
    )

    return counts


def summarize_duplicates(
    golden: pd.DataFrame,
) -> pd.DataFrame:
    """Check for duplicate customer messages and conversation roots."""

    message_duplicates = (
        golden["customer_message"]
        .duplicated()
        .sum()
    )

    id_duplicates = (
        golden["customer_tweet_id"]
        .duplicated()
        .sum()
    )

    conversation_duplicates = (
        golden["conversation_root"]
        .duplicated()
        .sum()
    )

    return pd.DataFrame([{
        "examples": len(golden),
        "duplicate_customer_messages": int(
            message_duplicates
        ),
        "duplicate_tweet_ids": int(
            id_duplicates
        ),
        "repeat_conversation_roots": int(
            conversation_duplicates
        ),
        "unique_conversations": int(
            golden["conversation_root"].nunique()
        ),
    }])


def add_message_flags(
    golden: pd.DataFrame,
) -> pd.DataFrame:
    """Add simple flags for potentially noisy examples."""

    result = golden.copy()

    text = (
        result["customer_message"]
        .fillna("")
        .astype(str)
    )

    result["message_length"] = text.str.len()

    result["very_short"] = (
        result["message_length"] < 20
    )

    result["very_long"] = (
        result["message_length"] > 180
    )

    result["has_question"] = text.str.contains(
        r"\?",
        regex=True,
        na=False,
    )

    result["has_order_id"] = text.str.contains(
        r"\b\d{3}[-\s]\d{5,}[-\s]\d{5,}\b",
        regex=True,
        na=False,
    )

    result["has_url"] = text.str.contains(
        r"https?://|www\.",
        regex=True,
        na=False,
    )

    return result

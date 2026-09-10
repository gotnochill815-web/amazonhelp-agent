
from __future__ import annotations

import pandas as pd


def build_parent_context(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each tweet, attach its immediate parent tweet text and metadata.

    This is used to provide annotators with minimal conversation context.
    """

    required = {
        "tweet_id",
        "in_response_to_tweet_id",
        "author_id",
        "text",
        "inbound",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    parent_lookup = df[
        [
            "tweet_id",
            "author_id",
            "text",
            "inbound",
        ]
    ].copy()

    parent_lookup = parent_lookup.rename(
        columns={
            "tweet_id": "parent_tweet_id",
            "author_id": "parent_author_id",
            "text": "parent_text",
            "inbound": "parent_inbound",
        }
    )

    result = df.merge(
        parent_lookup,
        left_on="in_response_to_tweet_id",
        right_on="parent_tweet_id",
        how="left",
    )

    return result


def format_context_row(row: pd.Series) -> str:
    """
    Create a compact human-readable context string.
    """

    parent_text = row.get("parent_text")

    if pd.isna(parent_text) or not str(parent_text).strip():
        return ""

    parent_inbound = row.get("parent_inbound")

    if bool(parent_inbound):
        speaker = "Previous customer message"
    else:
        speaker = "Previous AmazonHelp response"

    return f"{speaker}: {str(parent_text).strip()}"


def add_context_column(
    candidate_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach immediate parent context to candidate examples.
    """

    context_df = build_parent_context(raw_df)

    context_df = context_df[
        [
            "tweet_id",
            "parent_text",
            "parent_inbound",
        ]
    ].copy()

    result = candidate_df.merge(
        context_df,
        left_on="customer_tweet_id",
        right_on="tweet_id",
        how="left",
    )

    result["previous_context"] = result.apply(
        format_context_row,
        axis=1,
    )

    result = result.drop(columns=["tweet_id"])

    return result

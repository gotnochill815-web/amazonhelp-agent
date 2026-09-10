
from __future__ import annotations

import pandas as pd


def build_message_lookup(df: pd.DataFrame) -> dict:
    """
    Build a tweet_id -> row lookup.

    The tweet_id is stored only as the dictionary key.
    We never rely on a nested 'tweet_id' field.
    """

    required = {
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "created_at",
        "in_response_to_tweet_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    lookup = {}

    for _, row in df[
        [
            "tweet_id",
            "author_id",
            "inbound",
            "text",
            "created_at",
            "in_response_to_tweet_id",
        ]
    ].iterrows():

        tweet_id = int(row["tweet_id"])

        lookup[tweet_id] = {
            "author_id": row["author_id"],
            "inbound": bool(row["inbound"]),
            "text": row["text"],
            "created_at": row["created_at"],
            "parent_id": row["in_response_to_tweet_id"],
        }

    return lookup


def get_parent_context(
    tweet_id: int,
    lookup: dict,
    max_turns: int = 4,
) -> list[dict]:
    """
    Walk backwards from tweet_id and collect previous turns.

    The current tweet itself is excluded.
    """

    current_id = int(tweet_id)

    context_rows = []
    visited = set()

    for _ in range(max_turns):

        if current_id in visited:
            break

        visited.add(current_id)

        current = lookup.get(current_id)

        if current is None:
            break

        parent_id = current["parent_id"]

        if pd.isna(parent_id):
            break

        parent_id = int(parent_id)

        parent = lookup.get(parent_id)

        if parent is None:
            break

        context_rows.append({
            "tweet_id": parent_id,
            "inbound": parent["inbound"],
            "text": parent["text"],
            "created_at": parent["created_at"],
        })

        current_id = parent_id

    context_rows.reverse()

    return context_rows


def format_turn(row: dict) -> str:
    """Format one context turn."""

    speaker = (
        "Customer"
        if row["inbound"]
        else "AmazonHelp"
    )

    text = str(row["text"]).strip()

    return f"{speaker}: {text}"


def build_context_for_examples(
    golden: pd.DataFrame,
    raw_df: pd.DataFrame,
    max_context_turns: int = 4,
) -> pd.DataFrame:
    """
    Attach recent conversation context to Golden Set examples.
    """

    if "customer_tweet_id" not in golden.columns:
        raise ValueError(
            "Golden set must contain customer_tweet_id."
        )

    lookup = build_message_lookup(raw_df)

    result = golden.copy()

    contexts = []

    for tweet_id in result["customer_tweet_id"]:

        if pd.isna(tweet_id):
            contexts.append("")
            continue

        rows = get_parent_context(
            int(tweet_id),
            lookup,
            max_turns=max_context_turns,
        )

        context = "\n".join(
            format_turn(row)
            for row in rows
        )

        contexts.append(context)

    result["conversation_context"] = contexts

    result["context_available"] = (
        result["conversation_context"]
        .fillna("")
        .str.strip()
        .ne("")
    )

    return result

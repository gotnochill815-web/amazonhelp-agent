
from __future__ import annotations

from pathlib import Path

import pandas as pd


def assign_conversation_roots(
    pairs: pd.DataFrame,
    interaction_roots: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach reconstructed conversation roots to customer -> brand pairs.
    """

    root_map = interaction_roots.drop_duplicates(
        subset=["tweet_id"]
    ).copy()

    result = pairs.merge(
        root_map,
        left_on="customer_tweet_id",
        right_on="tweet_id",
        how="left",
    )

    result = result.drop(columns=["tweet_id"])

    return result


def temporal_conversation_split(
    pairs: pd.DataFrame,
    evaluation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split conversations chronologically.

    Entire conversations are kept in either the historical/retrieval
    partition or the evaluation partition.

    The cutoff is determined using the earliest customer message time
    associated with each conversation.
    """

    if "conversation_root" not in pairs.columns:
        raise ValueError(
            "pairs must contain conversation_root"
        )

    pairs = pairs.copy()

    pairs["customer_created_at"] = pd.to_datetime(
        pairs["customer_created_at"],
        errors="coerce",
        utc=True,
    )

    conversation_dates = (
        pairs.groupby("conversation_root")[
            "customer_created_at"
        ]
        .min()
        .sort_values()
    )

    cutoff_index = int(
        len(conversation_dates)
        * (1.0 - evaluation_fraction)
    )

    cutoff_index = min(
        max(cutoff_index, 1),
        len(conversation_dates) - 1,
    )

    cutoff = conversation_dates.iloc[cutoff_index]

    evaluation_roots = set(
        conversation_dates[
            conversation_dates >= cutoff
        ].index
    )

    retrieval = pairs[
        ~pairs["conversation_root"].isin(
            evaluation_roots
        )
    ].copy()

    evaluation = pairs[
        pairs["conversation_root"].isin(
            evaluation_roots
        )
    ].copy()

    return retrieval, evaluation


def summarize_split(
    retrieval: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    """Return split statistics."""

    return pd.DataFrame([
        {
            "partition": "retrieval",
            "rows": len(retrieval),
            "conversations": retrieval[
                "conversation_root"
            ].nunique(),
            "customers": retrieval[
                "customer_author_id"
            ].nunique(),
        },
        {
            "partition": "evaluation",
            "rows": len(evaluation),
            "conversations": evaluation[
                "conversation_root"
            ].nunique(),
            "customers": evaluation[
                "customer_author_id"
            ].nunique(),
        },
    ])

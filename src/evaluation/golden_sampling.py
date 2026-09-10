
from __future__ import annotations

import re

import pandas as pd


KEYWORD_GROUPS = {
    "delivery": [
        "delivery",
        "delivered",
        "package",
        "parcel",
        "shipping",
        "shipment",
        "tracking",
        "courier",
        "arrive",
        "arrived",
        "late",
    ],
    "order": [
        "order",
        "ordered",
        "cancel",
        "cancellation",
        "dispatch",
    ],
    "preorder": [
        "preorder",
        "pre-order",
        "pre ordered",
        "release day",
        "release date",
    ],
    "return": [
        "return",
        "returned",
        "replacement",
        "replace",
        "exchange",
    ],
    "refund": [
        "refund",
        "refunded",
        "money back",
    ],
    "payment": [
        "payment",
        "charged",
        "charge",
        "billing",
        "credit card",
        "debit card",
    ],
    "account_security": [
        "account",
        "password",
        "login",
        "log in",
        "locked",
        "phishing",
        "hacked",
        "unauthorized",
        "security",
    ],
    "prime": [
        "prime",
        "membership",
        "subscription",
    ],
    "digital": [
        "kindle",
        "echo",
        "alexa",
        "fire tv",
        "firetv",
        "prime video",
        "video",
        "app",
        "stream",
        "play",
        "playing",
        "freeze",
        "freezing",
    ],
    "complaint": [
        "terrible",
        "worst",
        "ridiculous",
        "disappointed",
        "useless",
        "angry",
        "pissed",
        "complaint",
        "customer service",
        "customer support",
        "escalate",
    ],
}


def add_weak_topic_signals(
    df: pd.DataFrame,
    text_column: str = "customer_message",
) -> pd.DataFrame:
    """
    Add exploratory weak topic signals.

    These signals are used ONLY for sampling the human annotation set.
    They are not treated as ground-truth labels.
    """

    result = df.copy()

    text = (
        result[text_column]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    for topic, keywords in KEYWORD_GROUPS.items():

        pattern = "|".join(
            re.escape(keyword).replace(
                r"\ ",
                r"\s+",
            )
            for keyword in keywords
        )

        result[f"signal_{topic}"] = text.str.contains(
            pattern,
            regex=True,
            na=False,
        )

    signal_columns = [
        column
        for column in result.columns
        if column.startswith("signal_")
    ]

    result["signal_count"] = result[
        signal_columns
    ].sum(axis=1)

    return result


def add_difficulty_signals(
    df: pd.DataFrame,
    text_column: str = "customer_message",
) -> pd.DataFrame:
    """
    Add simple difficulty indicators for candidate selection.
    """

    result = df.copy()

    text = (
        result[text_column]
        .fillna("")
        .astype(str)
    )

    result["text_length"] = text.str.len()

    result["has_question"] = text.str.contains(
        r"\?",
        regex=True,
        na=False,
    )

    result["has_number"] = text.str.contains(
        r"\d",
        regex=True,
        na=False,
    )

    result["very_short"] = result["text_length"] < 30

    result["very_long"] = result["text_length"] > 180

    result["likely_ambiguous"] = (
        (result["signal_count"] == 0)
        | (result["signal_count"] >= 3)
        | result["very_short"]
    )

    return result


def sample_candidate_pool(
    df: pd.DataFrame,
    n: int = 600,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a larger candidate pool for manual selection.

    The pool intentionally over-samples potentially difficult and
    multi-topic cases. It is not the final Golden Set.
    """

    df = df.copy()

    df = add_weak_topic_signals(df)
    df = add_difficulty_signals(df)

    pieces = []

    # General random coverage
    random_n = min(250, len(df))

    pieces.append(
        df.sample(
            n=random_n,
            random_state=random_state,
        )
    )

    # Potentially ambiguous cases
    ambiguous = df[
        df["likely_ambiguous"]
    ]

    if len(ambiguous):
        pieces.append(
            ambiguous.sample(
                n=min(150, len(ambiguous)),
                random_state=random_state + 1,
            )
        )

    # Multi-topic cases
    multi_topic = df[
        df["signal_count"] >= 2
    ]

    if len(multi_topic):
        pieces.append(
            multi_topic.sample(
                n=min(100, len(multi_topic)),
                random_state=random_state + 2,
            )
        )

    # Long/detail-rich cases
    detailed = df[
        df["very_long"]
    ]

    if len(detailed):
        pieces.append(
            detailed.sample(
                n=min(75, len(detailed)),
                random_state=random_state + 3,
            )
        )

    # Very short/follow-up cases
    short = df[
        df["very_short"]
    ]

    if len(short):
        pieces.append(
            short.sample(
                n=min(75, len(short)),
                random_state=random_state + 4,
            )
        )

    result = (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(
            subset=["customer_tweet_id"]
        )
        .sample(
            frac=1.0,
            random_state=random_state,
        )
        .reset_index(drop=True)
    )

    return result.head(
        min(n, len(result))
    )

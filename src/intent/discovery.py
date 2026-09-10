
import re
from typing import Iterable

import pandas as pd


def clean_customer_text(text: str) -> str:
    """
    Basic normalization for customer-support text.

    Removes:
    - @mentions
    - URLs
    - excessive whitespace

    Keeps the original wording otherwise.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_customer_messages(
    pairs: pd.DataFrame,
    min_chars: int = 10,
) -> pd.DataFrame:
    """
    Prepare customer messages for intent discovery.

    Returns one row per unique customer tweet.
    """

    required = {
        "customer_tweet_id",
        "customer_text",
        "customer_author_id",
    }

    missing = required - set(pairs.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    messages = pairs[
        [
            "customer_tweet_id",
            "customer_author_id",
            "customer_text",
        ]
    ].copy()

    messages["clean_text"] = (
        messages["customer_text"]
        .map(clean_customer_text)
    )

    # Remove empty / extremely short messages
    messages = messages[
        messages["clean_text"].str.len() >= min_chars
    ].copy()

    # Deduplicate exact repeated customer tweets
    messages = messages.drop_duplicates(
        subset=["customer_tweet_id"]
    )

    messages = messages.reset_index(drop=True)

    return messages


def sample_messages(
    messages: pd.DataFrame,
    n: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Draw a reproducible sample for intent discovery.
    """

    n = min(n, len(messages))

    return messages.sample(
        n=n,
        random_state=random_state,
    ).reset_index(drop=True)

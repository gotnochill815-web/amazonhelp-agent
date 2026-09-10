from typing import Dict
import pandas as pd


def filter_brand_tweets(
    df: pd.DataFrame,
    brand: str,
) -> pd.DataFrame:
    """Return tweets authored by a specific brand."""

    mask = (
        df["author_id"]
        .astype(str)
        .str.lower()
        .eq(brand.lower())
    )

    return df.loc[mask].copy()


def build_customer_brand_pairs(
    df: pd.DataFrame,
    brand: str,
) -> pd.DataFrame:
    """
    Build direct customer -> brand response pairs.

    Each row represents a brand reply that directly responds
    to a customer tweet.
    """

    brand_replies = df[
        (df["author_id"].astype(str).str.lower() == brand.lower())
        & (df["in_response_to_tweet_id"].notna())
    ].copy()

    customer_tweets = df[
        [
            "tweet_id",
            "author_id",
            "inbound",
            "text",
            "created_at",
        ]
    ].copy()

    customer_tweets = customer_tweets.rename(
        columns={
            "tweet_id": "customer_tweet_id",
            "author_id": "customer_author_id",
            "inbound": "customer_inbound",
            "text": "customer_text",
            "created_at": "customer_created_at",
        }
    )

    pairs = brand_replies.merge(
        customer_tweets,
        left_on="in_response_to_tweet_id",
        right_on="customer_tweet_id",
        how="inner",
    )

    pairs = pairs[pairs["customer_inbound"]].copy()

    pairs = pairs.rename(
        columns={
            "tweet_id": "brand_tweet_id",
            "text": "brand_response",
            "created_at": "brand_created_at",
        }
    )

    return pairs


def build_parent_map(df: pd.DataFrame) -> Dict[int, int]:
    """Create tweet_id -> parent_tweet_id mapping."""

    subset = df[
        ["tweet_id", "in_response_to_tweet_id"]
    ].dropna(subset=["in_response_to_tweet_id"])

    return dict(
        zip(
            subset["tweet_id"].astype(int),
            subset["in_response_to_tweet_id"].astype(int),
        )
    )


def find_conversation_root(
    tweet_id: int,
    parent_map: Dict[int, int],
    max_depth: int = 100,
) -> int:
    """Walk backwards through reply links to find the root tweet."""

    current = int(tweet_id)
    visited = set()

    for _ in range(max_depth):
        if current in visited:
            break

        visited.add(current)

        parent = parent_map.get(current)

        if parent is None:
            break

        current = parent

    return current

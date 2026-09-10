
import re
from collections import Counter
from typing import Iterable

import pandas as pd


DEFAULT_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "is",
    "it", "for", "on", "my", "me", "i", "you", "your", "this",
    "that", "with", "have", "has", "was", "be", "are", "but",
    "can", "do", "please", "hi", "hello", "from", "now",
    "get", "they", "when", "what", "how", "all", "just",
    "one", "there", "any", "will", "been", "still", "out",
    "it's", "don't", "i'm", "had", "want", "need", "know",
    "even", "day", "days", "today", "time",
}


def clean_customer_text(text: str) -> str:
    """Normalize a customer tweet while preserving the issue content."""

    if pd.isna(text):
        return ""

    text = str(text)

    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Twitter mentions
    text = re.sub(r"@\w+", " ", text)

    # HTML entities commonly found in the dataset
    text = re.sub(r"&amp;", "and", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;", "'", text)

    # Whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_english(text: str) -> list[str]:
    """Extract simple English word tokens for exploratory frequency analysis."""

    return re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())


def top_terms(
    texts: Iterable[str],
    n: int = 50,
    stopwords: set[str] | None = None,
) -> list[tuple[str, int]]:
    """Return the most common exploratory terms."""

    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    counter = Counter()

    for text in texts:
        tokens = tokenize_english(text)

        for token in tokens:
            if len(token) <= 2:
                continue

            if token in stopwords:
                continue

            counter[token] += 1

    return counter.most_common(n)


def message_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return useful message-length statistics."""

    lengths = df["clean_text"].str.len()

    return pd.DataFrame([{
        "count": len(lengths),
        "mean_chars": lengths.mean(),
        "median_chars": lengths.median(),
        "p90_chars": lengths.quantile(0.90),
        "p95_chars": lengths.quantile(0.95),
        "max_chars": lengths.max(),
    }])


def repeated_message_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Measure duplicate customer messages."""

    total = len(df)
    unique = df["clean_text"].nunique()
    duplicate_rows = total - unique

    return pd.DataFrame([{
        "total_messages": total,
        "unique_messages": unique,
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": duplicate_rows / total if total else 0,
    }])


def keyword_counts(
    df: pd.DataFrame,
    keywords: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Count messages containing any keyword associated with each
    exploratory topic.

    This is descriptive analysis, not ground-truth intent labelling.
    """

    rows = []

    texts = df["clean_text"].fillna("").str.lower()

    for category, terms in keywords.items():
        pattern = "|".join(
            re.escape(term.lower()) for term in terms
        )

        mask = texts.str.contains(
            pattern,
            regex=True,
            na=False,
        )

        rows.append({
            "topic": category,
            "matching_messages": int(mask.sum()),
            "match_rate": float(mask.mean()),
        })

    return pd.DataFrame(rows).sort_values(
        "matching_messages",
        ascending=False,
    ).reset_index(drop=True)

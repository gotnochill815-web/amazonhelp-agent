
from __future__ import annotations

import pandas as pd


def build_candidate_golden_set(
    evaluation_pairs: pd.DataFrame,
    n_examples: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create a candidate golden-set annotation sheet from the
    held-out evaluation partition.

    Sampling is performed at the customer-message level after the
    evaluation partition has already been separated by conversation.

    The resulting file is an annotation template, NOT ground truth.
    """

    required_columns = {
        "customer_tweet_id",
        "customer_author_id",
        "customer_text",
        "customer_created_at",
        "conversation_root",
    }

    missing = required_columns - set(evaluation_pairs.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    candidates = evaluation_pairs[
        [
            "customer_tweet_id",
            "customer_author_id",
            "customer_text",
            "customer_created_at",
            "conversation_root",
        ]
    ].copy()

    candidates = candidates.drop_duplicates(
        subset=["customer_tweet_id"]
    )

    candidates = candidates.dropna(
        subset=["customer_text"]
    )

    n_examples = min(
        n_examples,
        len(candidates),
    )

    sample = candidates.sample(
        n=n_examples,
        random_state=random_state,
    ).reset_index(drop=True)

    sample.insert(
        0,
        "example_id",
        range(1, len(sample) + 1),
    )

    sample["intent"] = ""
    sample["should_escalate"] = ""
    sample["difficulty"] = ""
    sample["annotation_notes"] = ""

    return sample


def validate_golden_annotations(
    golden: pd.DataFrame,
    valid_intents: set[str],
) -> None:
    """Validate manually completed golden-set annotations."""

    required_columns = {
        "example_id",
        "customer_message",
        "intent",
        "should_escalate",
        "difficulty",
    }

    missing = required_columns - set(golden.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    invalid_intents = set(
        golden["intent"].dropna().astype(str)
    ) - valid_intents - {""}

    if invalid_intents:
        raise ValueError(
            f"Invalid intents: {sorted(invalid_intents)}"
        )

    valid_escalation = {
        True,
        False,
        "True",
        "False",
        "",
    }

    invalid_escalation = set(
        golden["should_escalate"].astype(str)
    ) - {"True", "False", ""}

    if invalid_escalation:
        raise ValueError(
            "should_escalate must contain True/False values."
        )

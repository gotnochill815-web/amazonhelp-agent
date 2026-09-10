
from __future__ import annotations

import pandas as pd


def select_final_golden_candidates(
    candidate_pool: pd.DataFrame,
    n_total: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Select a balanced candidate set for the manually labelled Golden Set.

    This function does NOT assign ground-truth labels.

    Selection buckets:
    - regular coverage
    - difficult / ambiguous
    - escalation/context-heavy
    """

    df = candidate_pool.copy()

    # Ensure deterministic order before sampling.
    df = df.drop_duplicates(
        subset=["customer_tweet_id"]
    ).reset_index(drop=True)

    # -----------------------------------------
    # Bucket 1: regular coverage
    # -----------------------------------------

    regular = df[
        (~df["likely_ambiguous"])
        & (df["signal_count"] <= 1)
    ].copy()

    regular_n = min(120, len(regular))

    regular_sample = regular.sample(
        n=regular_n,
        random_state=random_state,
    ).copy()

    regular_sample["selection_reason"] = (
        "regular_coverage"
    )

    # -----------------------------------------
    # Bucket 2: difficult / ambiguous
    # -----------------------------------------

    difficult = df[
        (
            df["likely_ambiguous"]
            | (df["signal_count"] >= 2)
            | df["very_short"]
            | df["very_long"]
        )
        & (~df["customer_tweet_id"].isin(
            regular_sample["customer_tweet_id"]
        ))
    ].copy()

    difficult_n = min(40, len(difficult))

    difficult_sample = difficult.sample(
        n=difficult_n,
        random_state=random_state + 1,
    ).copy()

    difficult_sample["selection_reason"] = (
        "difficult_or_ambiguous"
    )

    # -----------------------------------------
    # Bucket 3: escalation/context-heavy
    # -----------------------------------------

    escalation_keywords = [
        "fraud",
        "phishing",
        "hacked",
        "unauthorized",
        "stolen",
        "locked",
        "account",
        "charged",
        "refund",
        "police",
        "complaint",
        "supervisor",
        "escalate",
        "legal",
        "lawyer",
        "scam",
        "scammed",
        "security",
        "personal information",
        "phone call",
    ]

    text = (
        df["customer_message"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    pattern = "|".join(
        keyword.replace(" ", r"\s+")
        for keyword in escalation_keywords
    )

    escalation_candidates = df[
        text.str.contains(
            pattern,
            regex=True,
            na=False,
        )
        | df["previous_context"].fillna("").str.len().gt(0)
    ].copy()

    escalation_candidates = escalation_candidates[
        ~escalation_candidates[
            "customer_tweet_id"
        ].isin(
            pd.concat(
                [
                    regular_sample["customer_tweet_id"],
                    difficult_sample["customer_tweet_id"],
                ]
            )
        )
    ]

    escalation_n = min(
        40,
        len(escalation_candidates)
    )

    escalation_sample = escalation_candidates.sample(
        n=escalation_n,
        random_state=random_state + 2,
    ).copy()

    escalation_sample["selection_reason"] = (
        "escalation_or_context_heavy"
    )

    # -----------------------------------------
    # Combine
    # -----------------------------------------

    result = pd.concat(
        [
            regular_sample,
            difficult_sample,
            escalation_sample,
        ],
        ignore_index=True,
    )

    # Remove accidental duplicates.
    result = result.drop_duplicates(
        subset=["customer_tweet_id"]
    )

    # If some buckets did not have enough examples,
    # fill remaining slots from the unused pool.
    remaining_n = n_total - len(result)

    if remaining_n > 0:

        remaining = df[
            ~df["customer_tweet_id"].isin(
                result["customer_tweet_id"]
            )
        ]

        fill = remaining.sample(
            n=min(remaining_n, len(remaining)),
            random_state=random_state + 3,
        ).copy()

        fill["selection_reason"] = (
            "additional_coverage"
        )

        result = pd.concat(
            [
                result,
                fill,
            ],
            ignore_index=True,
        )

    # Shuffle final set deterministically.
    result = result.sample(
        frac=1.0,
        random_state=random_state + 4,
    ).reset_index(drop=True)

    result.insert(
        0,
        "golden_example_id",
        range(1, len(result) + 1),
    )

    return result

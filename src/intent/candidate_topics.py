
from typing import Dict, List

import pandas as pd


CANDIDATE_TOPICS: Dict[str, List[str]] = {

    "delivery_delay": [
        "late",
        "delayed",
        "overdue",
        "days late",
        "still waiting",
        "taking too long",
        "not arrived yet",
        "was supposed to arrive",
    ],

    "delivery_not_received": [
        "marked delivered",
        "shows delivered",
        "says delivered",
        "not delivered",
        "didn't receive",
        "did not receive",
        "never received",
        "missing package",
        "wrong address",
        "wrong house",
        "left with neighbour",
        "left with neighbor",
    ],

    "order_status": [
        "order status",
        "where is my order",
        "where's my order",
        "tracking",
        "tracking number",
        "not dispatched",
        "hasn't shipped",
        "not shipped",
        "shipment status",
    ],

    "preorder_issue": [
        "pre-order",
        "preorder",
        "pre ordered",
        "preordered",
        "release date",
        "release day",
        "day one",
        "dayone",
    ],

    "returns_exchanges": [
        "return",
        "returned",
        "replacement",
        "replace",
        "exchange",
        "return label",
    ],

    "refund_issue": [
        "refund",
        "refunded",
        "money back",
        "refund hasn't",
        "refund not",
    ],

    "payment_billing": [
        "payment",
        "charged",
        "charge",
        "billing",
        "credit card",
        "debit card",
        "payment failed",
        "charged twice",
    ],

    "account_security": [
        "locked account",
        "account locked",
        "password",
        "login",
        "log in",
        "sign in",
        "hacked",
        "unauthorized",
        "security",
        "phishing",
    ],

    "prime_membership": [
        "amazon prime",
        "prime membership",
        "prime member",
        "prime subscription",
        "prime fee",
        "prime charge",
        "prime benefits",
    ],

    "product_digital_support": [
        "kindle",
        "echo",
        "alexa",
        "fire tv",
        "firetv",
        "prime video",
        "amazon video",
        "app",
        "can't play",
        "cant play",
        "won't play",
        "doesn't play",
        "doesnt play",
        "freezing",
        "buffering",
    ],

    "general_complaint_or_support": [
        "customer service",
        "customer support",
        "complaint",
        "terrible service",
        "worst service",
        "ridiculous",
        "disappointed",
        "useless",
        "help",
        "escalate",
    ],
}


def find_topic_matches(
    df: pd.DataFrame,
    topic_keywords: Dict[str, List[str]] = CANDIDATE_TOPICS,
    text_column: str = "clean_text",
) -> pd.DataFrame:
    """
    Find messages matching candidate topics.

    This is exploratory weak labeling only.
    It is NOT ground truth.
    """

    result = df.copy()

    text = (
        result[text_column]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    for topic, keywords in topic_keywords.items():

        pattern = "|".join(
            keyword.lower().replace(" ", r"\s+")
            for keyword in keywords
        )

        result[f"candidate_{topic}"] = text.str.contains(
            pattern,
            regex=True,
            na=False,
        )

    return result


def topic_match_counts(
    matched_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    topic_columns = [
        column
        for column in matched_df.columns
        if column.startswith("candidate_")
    ]

    for column in topic_columns:

        count = int(matched_df[column].sum())

        rows.append({
            "topic": column.replace("candidate_", ""),
            "matching_messages": count,
            "match_rate": count / len(matched_df),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "matching_messages",
            ascending=False,
        )
        .reset_index(drop=True)
    )

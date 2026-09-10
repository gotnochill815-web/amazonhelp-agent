
from pathlib import Path
import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


ROOT = Path("/content/hiver-amazonhelp-agent")
RESULTS = ROOT / "results"

RETRIEVAL_FILE = RESULTS / "retrieval_pairs_temporal.csv"
GOLDEN_FILE = RESULTS / "golden_set_eval.csv"


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Twitter mentions
    text = re.sub(r"@\w+", " ", text)

    # Keep words / unicode characters
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):
    text = clean_text(text)
    return text.split()


def load_data():

    retrieval = pd.read_csv(RETRIEVAL_FILE)
    golden = pd.read_csv(GOLDEN_FILE)

    required_retrieval = [
        "customer_text",
        "brand_response",
        "conversation_root",
        "customer_created_at",
    ]

    required_golden = [
        "golden_example_id",
        "customer_message",
        "intent",
    ]

    missing = [
        c for c in required_retrieval
        if c not in retrieval.columns
    ]

    if missing:
        raise ValueError(
            f"Missing retrieval columns: {missing}"
        )

    missing = [
        c for c in required_golden
        if c not in golden.columns
    ]

    if missing:
        raise ValueError(
            f"Missing Golden Set columns: {missing}"
        )

    return retrieval, golden


def build_index(retrieval):

    retrieval = retrieval.copy()

    retrieval["clean_customer_text"] = (
        retrieval["customer_text"]
        .map(clean_text)
    )

    retrieval = retrieval[
        retrieval["clean_customer_text"].str.len() > 0
    ].reset_index(drop=True)

    tokenized_documents = [
        tokenize(text)
        for text in retrieval["clean_customer_text"]
    ]

    bm25 = BM25Okapi(tokenized_documents)

    return retrieval, bm25


def retrieve(
    query,
    retrieval,
    bm25,
    top_k=5,
):

    tokens = tokenize(query)

    scores = bm25.get_scores(tokens)

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = retrieval.iloc[top_indices].copy()

    results["retrieval_score"] = scores[top_indices]

    return results.reset_index(drop=True)


def main():

    print("=" * 70)
    print("BM25 RETRIEVAL BASELINE")
    print("=" * 70)

    retrieval, golden = load_data()

    print(
        f"Historical retrieval rows: {len(retrieval)}"
    )

    print(
        f"Golden Set rows:           {len(golden)}"
    )

    retrieval, bm25 = build_index(
        retrieval
    )

    print(
        f"Usable retrieval rows:     {len(retrieval)}"
    )

    all_rows = []

    for _, row in golden.iterrows():

        retrieved = retrieve(
            query=row["customer_message"],
            retrieval=retrieval,
            bm25=bm25,
            top_k=5,
        )

        for rank, (_, result) in enumerate(
            retrieved.iterrows(),
            start=1,
        ):

            all_rows.append({

                "golden_example_id":
                    row["golden_example_id"],

                "golden_intent":
                    row["intent"],

                "customer_message":
                    row["customer_message"],

                "rank":
                    rank,

                "retrieval_score":
                    result["retrieval_score"],

                "retrieved_customer_text":
                    result["customer_text"],

                "retrieved_brand_response":
                    result["brand_response"],

                "retrieved_conversation_root":
                    result["conversation_root"],

                "retrieved_customer_created_at":
                    result["customer_created_at"],
            })

    results = pd.DataFrame(all_rows)

    top5_output = (
        RESULTS /
        "bm25_retrieval_top5.csv"
    )

    results.to_csv(
        top5_output,
        index=False,
    )

    top1 = results[
        results["rank"] == 1
    ].copy()

    top1_output = (
        RESULTS /
        "bm25_retrieval_top1.csv"
    )

    top1.to_csv(
        top1_output,
        index=False,
    )

    print("\nSaved:")
    print(top5_output)
    print(top1_output)


if __name__ == "__main__":
    main()

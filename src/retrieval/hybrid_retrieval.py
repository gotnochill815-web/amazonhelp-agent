
from pathlib import Path
import re

import numpy as np
import pandas as pd

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


ROOT = Path("/content/hiver-amazonhelp-agent")

RETRIEVAL_PATH = (
    ROOT / "results" / "retrieval_pairs_temporal_gold_excluded.csv"
)

GOLD_PATH = (
    ROOT / "results" / "golden_set_v2_rebuilt_context.csv"
)

TOP_K_BM25 = 50
TOP_K_DENSE = 50
TOP_K_HYBRID = 50
TOP_K_FINAL = 5

ALPHA = 0.5

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str):
    return normalize_text(text).split()


def minmax_normalize(scores):
    scores = np.asarray(scores, dtype=float)

    if len(scores) == 0:
        return scores

    mn = scores.min()
    mx = scores.max()

    if np.isclose(mx, mn):
        return np.ones_like(scores)

    return (scores - mn) / (mx - mn)


def load_data():
    retrieval = pd.read_csv(
        RETRIEVAL_PATH,
        dtype=str,
        keep_default_na=False
    )

    gold = pd.read_csv(
        GOLD_PATH,
        dtype=str,
        keep_default_na=False
    )

    required_retrieval = {
        "conversation_root",
        "customer_tweet_id",
        "customer_text",
        "brand_response",
    }

    missing = required_retrieval - set(retrieval.columns)

    if missing:
        raise ValueError(
            f"Retrieval data missing columns: {sorted(missing)}"
        )

    if "customer_text" not in gold.columns:
        raise ValueError("Golden Set missing customer_text")

    return retrieval, gold


def build_bm25(retrieval):
    tokenized = [
        tokenize(text)
        for text in retrieval["customer_text"].tolist()
    ]

    return BM25Okapi(tokenized)


def retrieve_bm25(query, bm25, top_k=TOP_K_BM25):
    scores = np.asarray(
        bm25.get_scores(tokenize(query)),
        dtype=float
    )

    k = min(top_k, len(scores))
    indices = np.argsort(scores)[::-1][:k]

    return {
        int(idx): float(scores[idx])
        for idx in indices
    }


def retrieve_dense(
    query_embedding,
    retrieval_embeddings,
    top_k=TOP_K_DENSE
):
    scores = retrieval_embeddings @ query_embedding

    k = min(top_k, len(scores))
    indices = np.argsort(scores)[::-1][:k]

    return {
        int(idx): float(scores[idx])
        for idx in indices
    }


def hybrid_candidates(
    bm25_results,
    dense_results,
    alpha=ALPHA,
    top_k=TOP_K_HYBRID,
):
    candidate_ids = sorted(
        set(bm25_results) | set(dense_results)
    )

    bm25_values = np.array(
        [bm25_results.get(i, 0.0) for i in candidate_ids],
        dtype=float
    )

    dense_values = np.array(
        [dense_results.get(i, 0.0) for i in candidate_ids],
        dtype=float
    )

    bm25_norm = minmax_normalize(bm25_values)
    dense_norm = minmax_normalize(dense_values)

    rows = []

    for j, idx in enumerate(candidate_ids):

        score = (
            alpha * bm25_norm[j]
            + (1.0 - alpha) * dense_norm[j]
        )

        rows.append(
            {
                "index": int(idx),
                "bm25_score": float(bm25_values[j]),
                "dense_score": float(dense_values[j]),
                "bm25_norm": float(bm25_norm[j]),
                "dense_norm": float(dense_norm[j]),
                "hybrid_score": float(score),
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "hybrid_score",
        ascending=False
    ).head(top_k)

    return result.reset_index(drop=True)


def rerank(
    query,
    candidates,
    retrieval,
    reranker,
):
    # IMPORTANT:
    # Rank the historical customer problem, not the response.
    pairs = []

    for idx in candidates["index"]:
        historical_query = retrieval.iloc[int(idx)]["customer_text"]

        pairs.append(
            [
                query,
                historical_query,
            ]
        )

    scores = reranker.predict(
        pairs,
        batch_size=64,
        show_progress_bar=False,
    )

    result = candidates.copy()
    result["cross_encoder_score"] = scores

    result = result.sort_values(
        "cross_encoder_score",
        ascending=False
    ).head(TOP_K_FINAL)

    return result.reset_index(drop=True)


def run():

    retrieval, gold = load_data()

    print(f"Retrieval rows: {len(retrieval):,}")
    print(f"Golden rows: {len(gold):,}")

    # --------------------------------------------------------
    # Leakage safety
    # --------------------------------------------------------
    retrieval_roots = set(
        retrieval["conversation_root"].astype(str)
    )

    gold_roots = set(
        gold["conversation_root"].astype(str)
    )

    overlap = retrieval_roots & gold_roots

    if overlap:
        raise RuntimeError(
            f"Golden conversation leakage detected: {len(overlap)}"
        )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------
    print("Building BM25...")
    bm25 = build_bm25(retrieval)

    print("Loading dense encoder...")
    dense_model = SentenceTransformer(
        DENSE_MODEL_NAME
    )

    print("Encoding retrieval corpus...")

    retrieval_embeddings = dense_model.encode(
        retrieval["customer_text"].tolist(),
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    print("Loading CrossEncoder...")
    reranker = CrossEncoder(
        RERANKER_MODEL_NAME
    )

    # --------------------------------------------------------
    # Output rows
    # --------------------------------------------------------
    top50_rows = []
    top5_rows = []

    # --------------------------------------------------------
    # Golden evaluation
    # --------------------------------------------------------
    for i, row in gold.iterrows():

        query = row["customer_text"]

        query_embedding = dense_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype("float32")

        bm25_results = retrieve_bm25(
            query,
            bm25,
            TOP_K_BM25,
        )

        dense_results = retrieve_dense(
            query_embedding,
            retrieval_embeddings,
            TOP_K_DENSE,
        )

        hybrid = hybrid_candidates(
            bm25_results,
            dense_results,
            alpha=ALPHA,
            top_k=TOP_K_HYBRID,
        )

        reranked = rerank(
            query,
            hybrid,
            retrieval,
            reranker,
        )

        # ----------------------------------------------------
        # Top-50
        # ----------------------------------------------------
        for rank, candidate in hybrid.iterrows():

            idx = int(candidate["index"])
            r = retrieval.iloc[idx]

            top50_rows.append(
                {
                    "golden_example_id": row["golden_example_id"],
                    "query": query,
                    "rank": rank + 1,
                    "customer_tweet_id": r["customer_tweet_id"],
                    "conversation_root": r["conversation_root"],
                    "historical_customer_text": r["customer_text"],
                    "brand_response": r["brand_response"],
                    "bm25_score": candidate["bm25_score"],
                    "dense_score": candidate["dense_score"],
                    "bm25_norm": candidate["bm25_norm"],
                    "dense_norm": candidate["dense_norm"],
                    "hybrid_score": candidate["hybrid_score"],
                }
            )

        # ----------------------------------------------------
        # Top-5 after CrossEncoder
        # ----------------------------------------------------
        for rank, candidate in reranked.iterrows():

            idx = int(candidate["index"])
            r = retrieval.iloc[idx]

            top5_rows.append(
                {
                    "golden_example_id": row["golden_example_id"],
                    "query": query,
                    "rank": rank + 1,
                    "customer_tweet_id": r["customer_tweet_id"],
                    "conversation_root": r["conversation_root"],
                    "historical_customer_text": r["customer_text"],
                    "brand_response": r["brand_response"],
                    "cross_encoder_score": candidate[
                        "cross_encoder_score"
                    ],
                    "hybrid_score": candidate["hybrid_score"],
                    "bm25_score": candidate["bm25_score"],
                    "dense_score": candidate["dense_score"],
                }
            )

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(gold)}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    out50 = ROOT / "results" / "hybrid_retrieval_top50.csv"
    out5 = ROOT / "results" / "hybrid_retrieval_top5.csv"

    pd.DataFrame(top50_rows).to_csv(
        out50,
        index=False
    )

    pd.DataFrame(top5_rows).to_csv(
        out5,
        index=False
    )

    print()
    print("Saved:")
    print(out50)
    print(out5)


if __name__ == "__main__":
    run()

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.generate import generate_with_openai


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RETRIEVAL_FILE = (
    ROOT / "results" / "retrieval_pairs_temporal_gold_excluded.csv"
)

DEMO_RETRIEVAL_FILE = (
    ROOT / "results" / "dense_comparison_top5.csv"
)

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_MIN_SCORE = 0.70


INTENTS = [
    "delivery_status",
    "delivery_problem",
    "order_issue",
    "preorder_issue",
    "returns_replacements",
    "refund_issue",
    "payment_billing",
    "account_security",
    "prime_membership",
    "digital_product_support",
    "general_support",
]


HIGH_RISK_INTENTS = {
    "account_security",
    "payment_billing",
}


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def rule_based_intent(text: str) -> Optional[str]:
    """
    High-precision rules for obvious support boundaries.

    Returns None when semantic classification should handle the query.
    """

    q = normalize(text)

    security_terms = [
        "locked account",
        "account locked",
        "hacked",
        "compromised",
        "fraud",
        "unauthorized",
        "stolen account",
        "security",
        "someone accessed",
    ]

    refund_terms = [
        "refund",
        "money back",
        "refunded",
        "refund not",
        "refund hasn't",
        "refund has not",
    ]

    return_terms = [
        "return this",
        "want to return",
        "return an item",
        "return item",
        "send it back",
        "replacement",
        "replace this",
        "exchange this",
    ]

    digital_terms = [
        "alexa",
        "kindle",
        "fire tv",
        "fire tablet",
        "prime video",
        "audible",
        "ebook",
        "e-book",
        "digital",
    ]

    membership_terms = [
        "amazon prime",
        "prime membership",
        "prime subscription",
        "prime member",
        "prime trial",
    ]

    payment_terms = [
        "charged",
        "charge",
        "billing",
        "payment",
        "credit card",
        "debit card",
        "card charged",
    ]

    preorder_terms = [
        "preorder",
        "pre-order",
    ]

    if any(term in q for term in security_terms):
        return "account_security"

    if any(term in q for term in refund_terms):
        return "refund_issue"

    if any(term in q for term in return_terms):
        return "returns_replacements"

    if any(term in q for term in digital_terms):
        return "digital_product_support"

    if any(term in q for term in membership_terms):
        return "prime_membership"

    if any(term in q for term in payment_terms):
        return "payment_billing"

    if any(term in q for term in preorder_terms):
        return "preorder_issue"

    return None


class MiniLMIntentClassifier:
    """
    Lightweight centroid classifier trained from the 152 genuinely
    human-labelled Gold examples.

    The fixed holdout benchmark remains separate from operational inference.
    """

    def __init__(
        self,
        gold_path: Path | None = None,
        model_name: str = DENSE_MODEL,
    ):
        self.gold_path = (
            gold_path
            or ROOT
            / "results"
            / "golden_set_human_review_completed.csv"
        )

        self.encoder = SentenceTransformer(model_name)
        self.centroids = {}
        self.intent_counts = {}

        self._fit()

    def _fit(self) -> None:
        if not self.gold_path.exists():
            raise FileNotFoundError(
                f"Human Gold file not found: {self.gold_path}"
            )

        df = pd.read_csv(self.gold_path)

        required = {
            "customer_message",
            "human_intent",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing columns in Gold file: {sorted(missing)}"
            )

        df = df.dropna(
            subset=["customer_message", "human_intent"]
        ).copy()

        texts = df["customer_message"].astype(str).tolist()
        labels = df["human_intent"].astype(str).tolist()

        embeddings = self.encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        for intent in sorted(set(labels)):
            idx = [
                i
                for i, label in enumerate(labels)
                if label == intent
            ]

            centroid = np.asarray(
                embeddings[idx]
            ).mean(axis=0)

            norm = np.linalg.norm(centroid)

            if norm > 0:
                centroid = centroid / norm

            self.centroids[intent] = centroid
            self.intent_counts[intent] = len(idx)

    def predict(self, text: str) -> dict:
        rule_intent = rule_based_intent(text)

        if rule_intent is not None:
            return {
                "intent": rule_intent,
                "score": 1.0,
                "second_best_intent": None,
                "second_best_score": 0.0,
                "margin": 1.0,
                "confidence_band": "high",
                "source": "rule",
            }

        embedding = self.encoder.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = []

        for intent, centroid in self.centroids.items():
            score = float(
                np.dot(embedding, centroid)
            )
            scores.append(
                (intent, score)
            )

        scores.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        best_intent, best_score = scores[0]

        if len(scores) > 1:
            second_intent, second_score = scores[1]
        else:
            second_intent, second_score = None, 0.0

        margin = best_score - second_score

        if best_score >= 0.70 and margin >= 0.10:
            band = "high"
        elif best_score >= 0.55:
            band = "medium"
        else:
            band = "low"

        return {
            "intent": best_intent,
            "score": float(best_score),
            "second_best_intent": second_intent,
            "second_best_score": float(second_score),
            "margin": float(margin),
            "confidence_band": band,
            "source": "minilm",
        }


class DenseRetriever:
    """
    Dense historical retrieval over a CSV of customer -> AmazonHelp
    interactions.

    For the committed evaluation artifact, retrieval evidence is already
    computed. For a live run, the class can index a temporal-safe corpus.
    """

    def __init__(
        self,
        corpus_path: Path | None = None,
        model_name: str = DENSE_MODEL,
    ):
        self.corpus_path = (
            corpus_path
            or DEFAULT_RETRIEVAL_FILE
        )

        if not self.corpus_path.exists():
            if DEMO_RETRIEVAL_FILE.exists():
                self.corpus_path = DEMO_RETRIEVAL_FILE
            else:
                raise FileNotFoundError(
                    "No retrieval corpus found. Expected either:\n"
                    f"  {DEFAULT_RETRIEVAL_FILE}\n"
                    f"  {DEMO_RETRIEVAL_FILE}"
                )

        self.encoder = SentenceTransformer(model_name)
        self.df = pd.read_csv(self.corpus_path)

        self.customer_col = self._find_column(
            [
                "customer_message",
                "historical_customer_text",
                "customer_text",
                "text",
            ]
        )

        self.response_col = self._find_column(
            [
                "brand_response",
                "historical_response",
                "response",
            ],
            required=False,
        )

        texts = (
            self.df[self.customer_col]
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.embeddings = self.encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _find_column(
        self,
        candidates: list[str],
        required: bool = True,
    ) -> Optional[str]:

        for col in candidates:
            if col in self.df.columns:
                return col

        if required:
            raise ValueError(
                "Could not find a supported customer-text column. "
                f"Available columns: {list(self.df.columns)}"
            )

        return None

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = np.dot(
            self.embeddings,
            query_embedding,
        )

        indices = np.argsort(
            -scores
        )[:top_k]

        results = []

        for rank, idx in enumerate(indices, start=1):
            row = self.df.iloc[int(idx)]

            item = {
                "rank": rank,
                "dense_score": float(scores[idx]),
                "historical_customer_text": str(
                    row[self.customer_col]
                ),
            }

            if self.response_col is not None:
                item["brand_response"] = str(
                    row[self.response_col]
                )

            for col in [
                "customer_tweet_id",
                "tweet_id",
                "conversation_root",
                "created_at",
            ]:
                if col in row.index:
                    item[col] = row[col]

            results.append(item)

        return results


class AmazonHelpAgent:

    def __init__(
        self,
        retrieval_path: Path | None = None,
        gold_path: Path | None = None,
    ):
        self.intent_classifier = MiniLMIntentClassifier(
            gold_path=gold_path
        )

        self.retriever = DenseRetriever(
            corpus_path=retrieval_path
        )

    def assess_evidence(
        self,
        evidence: list[dict],
    ) -> dict:

        if not evidence:
            return {
                "retrieval_available": False,
                "top_score": 0.0,
                "evidence_sufficient": False,
            }

        top_score = float(
            max(
                float(item.get("dense_score", 0.0))
                for item in evidence
            )
        )

        return {
            "retrieval_available": True,
            "top_score": top_score,
            "evidence_sufficient": (
                top_score >= DENSE_MIN_SCORE
            ),
        }

    def decide_escalation(
        self,
        intent_result: dict,
        evidence_assessment: dict,
        generated,
    ) -> tuple[bool, list[str]]:

        reasons = []

        intent = intent_result["intent"]

        if intent in HIGH_RISK_INTENTS:
            reasons.append(
                "high-risk support intent"
            )

        if (
            intent_result["confidence_band"]
            in {"low", "medium"}
        ):
            reasons.append(
                "intent classification has moderate or low uncertainty"
            )

        if not evidence_assessment[
            "evidence_sufficient"
        ]:
            reasons.append(
                "historical evidence is insufficient"
            )

        if generated is not None:
            if generated.needs_more_information:
                reasons.append(
                    "additional customer information is required"
                )

        should_escalate = len(reasons) > 0

        return should_escalate, reasons

    def run(
        self,
        query: str,
        top_k: int = 5,
        model: str | None = None,
    ) -> dict:

        intent_result = (
            self.intent_classifier.predict(query)
        )

        evidence = self.retriever.search(
            query,
            top_k=top_k,
        )

        evidence_assessment = (
            self.assess_evidence(evidence)
        )

        generated = None

        # Generate only when the retrieval stage has evidence.
        if evidence_assessment[
            "evidence_sufficient"
        ]:

            generated = generate_with_openai(
                query=query,
                evidence=evidence,
                model=model,
            )

        should_escalate, escalation_reasons = (
            self.decide_escalation(
                intent_result=intent_result,
                evidence_assessment=evidence_assessment,
                generated=generated,
            )
        )

        if generated is None:
            draft_response = (
                "I couldn't find sufficiently strong historical evidence "
                "to safely draft a response. Please route this request "
                "to a support specialist."
            )
        else:
            draft_response = generated.draft_response

        return {
            "query": query,
            "intent": intent_result,
            "retrieved_evidence": evidence,
            "evidence_assessment": evidence_assessment,
            "draft_response": draft_response,
            "grounded_claim": (
                generated.grounded_claim
                if generated is not None
                else ""
            ),
            "needs_more_information": (
                generated.needs_more_information
                if generated is not None
                else True
            ),
            "should_escalate": should_escalate,
            "escalation_reasons": escalation_reasons,
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="AmazonHelp end-to-end support agent"
    )

    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--model",
        default=None,
    )

    args = parser.parse_args()

    agent = AmazonHelpAgent()

    result = agent.run(
        query=args.query,
        top_k=args.top_k,
        model=args.model,
    )

    print("\n" + "=" * 72)
    print("AMAZONHELP AI SUPPORT AGENT")
    print("=" * 72)

    print("\nCUSTOMER")
    print(result["query"])

    print("\nINTENT")
    print(result["intent"])

    print("\nTOP EVIDENCE")
    for item in result["retrieved_evidence"]:
        print(
            f"rank={item['rank']} "
            f"score={item['dense_score']:.4f}"
        )
        print(
            item["historical_customer_text"][:300]
        )
        print()

    print("\nDRAFT RESPONSE")
    print(result["draft_response"])

    print("\nGROUNDING")
    print(result["grounded_claim"])

    print(
        "\nNEEDS MORE INFORMATION:",
        result["needs_more_information"],
    )

    print(
        "\nSHOULD ESCALATE:",
        result["should_escalate"],
    )

    print(
        "\nESCALATION REASONS:",
        "; ".join(
            result["escalation_reasons"]
        )
        if result["escalation_reasons"]
        else "none",
    )


if __name__ == "__main__":
    main()

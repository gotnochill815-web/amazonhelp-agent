from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


INTENT_DEFINITIONS = {
    "delivery_status": (
        "Customer wants to know where an order or package is, "
        "when it will arrive, tracking status, or expected delivery time."
    ),

    "delivery_problem": (
        "A delivery has gone wrong, such as being late, missed, "
        "not delivered, delivered incorrectly, damaged in transit, "
        "or problems with the carrier."
    ),

    "order_issue": (
        "Problem with an order itself, including order placement, "
        "cancellation, modification, confirmation, address, or "
        "unexpected order behavior that is not primarily delivery."
    ),

    "preorder_issue": (
        "Issue specifically involving a pre-order, including "
        "pre-order timing, availability, release, or fulfillment."
    ),

    "returns_replacements": (
        "Customer wants to return, replace, exchange, or otherwise "
        "resolve an item through a return or replacement workflow."
    ),

    "refund_issue": (
        "Customer is asking about a refund, refund timing, missing "
        "refund, refund amount, or reimbursement after a transaction."
    ),

    "payment_billing": (
        "Payment or billing issue, including charges, payment methods, "
        "billing errors, invoices, or unrecognized charges."
    ),

    "account_security": (
        "Account access, compromised account, suspicious login, "
        "fraud/security, password, or account-protection issue."
    ),

    "prime_membership": (
        "Questions or problems specifically about Amazon Prime "
        "membership, Prime benefits, subscription, or eligibility."
    ),

    "digital_product_support": (
        "Support for digital products or devices/services such as "
        "Alexa, Echo, Kindle, digital content, streaming, or device features."
    ),

    "general_support": (
        "General Amazon support where the message does not provide "
        "enough information to assign a more specific intent."
    ),
}


@dataclass
class IntentPrediction:
    intent: str
    score: float
    second_best_intent: str
    second_best_score: float
    margin: float
    confidence_band: str


class MiniLMIntentClassifier:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

        self.centroids: Dict[str, np.ndarray] = {}

        self.intents: List[str] = []

    def fit(
        self,
        messages: List[str],
        labels: List[str],
    ) -> None:

        if len(messages) != len(labels):
            raise ValueError(
                "messages and labels must have equal length"
            )

        embeddings = self.model.encode(
            messages,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        labels_array = np.asarray(labels)

        self.intents = sorted(
            set(labels_array.tolist())
        )

        for intent in self.intents:

            mask = labels_array == intent

            if not np.any(mask):
                continue

            centroid = embeddings[mask].mean(
                axis=0
            )

            centroid = centroid / (
                np.linalg.norm(centroid)
                + 1e-12
            )

            self.centroids[intent] = centroid

    def predict(
        self,
        message: str,
    ) -> IntentPrediction:

        if not self.centroids:
            raise RuntimeError(
                "Classifier has not been fitted."
            )

        if not isinstance(message, str) or not message.strip():
            raise ValueError(
                "message must be a non-empty string"
            )

        embedding = self.model.encode(
            [message],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        scores = {
            intent: float(
                embedding @ centroid
            )
            for intent, centroid
            in self.centroids.items()
        }

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        best_intent, best_score = ranked[0]

        if len(ranked) > 1:
            second_intent, second_score = ranked[1]
        else:
            second_intent = "general_support"
            second_score = 0.0

        margin = (
            best_score - second_score
        )

        if margin >= 0.10:
            confidence_band = "high"
        elif margin >= 0.05:
            confidence_band = "medium"
        else:
            confidence_band = "low"

        return IntentPrediction(
            intent=best_intent,
            score=best_score,
            second_best_intent=second_intent,
            second_best_score=second_score,
            margin=margin,
            confidence_band=confidence_band,
        )

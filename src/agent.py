
"""
Evidence-aware AmazonHelp support agent.

Pipeline:
1. Dense retrieval over historical AmazonHelp customer/brand interactions.
2. Evidence-quality assessment.
3. Grounded response generation is only allowed when evidence is sufficient.
4. Escalation is explicit and separate from intent classification.

Important:
- Dense similarity is NOT treated as calibrated answer confidence.
- The dense score is only a retrieval-quality signal.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Calibrated engineering gate.
# This is NOT a probability.
DENSE_MIN_SCORE = 0.70

DEFAULT_TOP_K = 5

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RETRIEVAL_PATH = (
    REPO_ROOT
    / "results"
    / "retrieval_pairs_temporal_gold_excluded.csv"
)


@dataclass
class RetrievalEvidence:
    rank: int
    dense_score: float
    customer_message: str
    historical_response: str
    conversation_root: Optional[str]


@dataclass
class EvidenceAssessment:
    sufficient: bool
    strongest_score: float
    evidence_count: int
    reason: str


@dataclass
class AgentResult:
    query: str
    evidence: List[Dict[str, Any]]
    assessment: Dict[str, Any]


class DenseRetriever:
    """Dense retriever over historical AmazonHelp interactions."""

    def __init__(
        self,
        retrieval_path: str | Path = DEFAULT_RETRIEVAL_PATH,
        model_name: str = DEFAULT_MODEL,
    ) -> None:

        self.retrieval_path = Path(retrieval_path)

        if not self.retrieval_path.exists():
            raise FileNotFoundError(
                f"Retrieval corpus not found: {self.retrieval_path}"
            )

        self.df = pd.read_csv(self.retrieval_path)

        required_columns = {
            "customer_text",
            "brand_response",
            "conversation_root",
        }

        missing = required_columns - set(self.df.columns)

        if missing:
            raise ValueError(
                f"Retrieval corpus missing columns: {sorted(missing)}"
            )

        self.df["customer_text"] = (
            self.df["customer_text"]
            .fillna("")
            .astype(str)
        )

        self.df["brand_response"] = (
            self.df["brand_response"]
            .fillna("")
            .astype(str)
        )

        self.model = SentenceTransformer(model_name)

        self.embeddings = self.model.encode(
            self.df["customer_text"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[RetrievalEvidence]:

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        scores = self.embeddings @ query_embedding

        top_k = min(top_k, len(scores))

        indices = np.argsort(-scores)[:top_k]

        evidence: List[RetrievalEvidence] = []

        for rank, idx in enumerate(indices, start=1):
            row = self.df.iloc[int(idx)]

            evidence.append(
                RetrievalEvidence(
                    rank=rank,
                    dense_score=float(scores[idx]),
                    customer_message=str(row["customer_text"]),
                    historical_response=str(row["brand_response"]),
                    conversation_root=str(
                        row["conversation_root"]
                    ),
                )
            )

        return evidence


class AmazonHelpAgent:
    """
    Evidence-aware support agent.

    The current stage returns retrieval + evidence assessment.
    Response generation is intentionally separate so that we can
    evaluate grounding before allowing generation to auto-handle.
    """

    def __init__(
        self,
        retrieval_path: str | Path = DEFAULT_RETRIEVAL_PATH,
        model_name: str = DEFAULT_MODEL,
        min_dense_score: float = DENSE_MIN_SCORE,
    ) -> None:

        if not 0.0 <= min_dense_score <= 1.0:
            raise ValueError(
                "min_dense_score must be between 0 and 1"
            )

        self.min_dense_score = min_dense_score

        self.retriever = DenseRetriever(
            retrieval_path=retrieval_path,
            model_name=model_name,
        )

    def assess_evidence(
        self,
        evidence: List[RetrievalEvidence],
    ) -> EvidenceAssessment:

        if not evidence:
            return EvidenceAssessment(
                sufficient=False,
                strongest_score=0.0,
                evidence_count=0,
                reason="No historical evidence was retrieved.",
            )

        strongest_score = max(
            item.dense_score
            for item in evidence
        )

        qualifying = [
            item
            for item in evidence
            if item.dense_score >= self.min_dense_score
        ]

        if not qualifying:
            return EvidenceAssessment(
                sufficient=False,
                strongest_score=strongest_score,
                evidence_count=0,
                reason=(
                    "Retrieved interactions are below the calibrated "
                    "dense-score quality gate. Do not rely on them "
                    "for an unsupported automated resolution."
                ),
            )

        return EvidenceAssessment(
            sufficient=True,
            strongest_score=strongest_score,
            evidence_count=len(qualifying),
            reason=(
                f"{len(qualifying)} retrieved interaction(s) meet "
                f"the dense retrieval quality gate of "
                f"{self.min_dense_score:.2f}. Evidence still requires "
                "semantic grounding review before auto-resolution."
            ),
        )

    def analyze(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> AgentResult:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        evidence = self.retriever.retrieve(
            query=query.strip(),
            top_k=top_k,
        )

        assessment = self.assess_evidence(evidence)

        return AgentResult(
            query=query.strip(),
            evidence=[asdict(item) for item in evidence],
            assessment=asdict(assessment),
        )


__all__ = [
    "AmazonHelpAgent",
    "DenseRetriever",
    "RetrievalEvidence",
    "EvidenceAssessment",
    "AgentResult",
    "DENSE_MIN_SCORE",
]

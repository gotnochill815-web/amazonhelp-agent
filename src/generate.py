from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable


@dataclass
class GeneratedResponse:
    draft_response: str
    grounded_claim: str
    needs_more_information: bool


SYSTEM_PROMPT = """
You are an Amazon customer-support reply drafter.

Use ONLY the historical AmazonHelp responses supplied as evidence.

Rules:
1. Do not invent policies, refund amounts, eligibility, delivery promises,
   account actions, or commitments.
2. Do not claim that an action has already been taken.
3. Ask for missing information when the evidence does not support a concrete answer.
4. Keep the response concise and customer-facing.
5. Ground every substantive claim in the supplied historical evidence.
6. Return valid JSON with:
   draft_response
   grounded_claim
   needs_more_information
"""


def build_evidence(evidence: Iterable[dict]) -> str:
    blocks = []

    for i, item in enumerate(evidence, start=1):
        customer = str(
            item.get("historical_customer_text")
            or item.get("customer_message")
            or ""
        ).strip()

        response = str(
            item.get("brand_response")
            or item.get("historical_response")
            or ""
        ).strip()

        if not customer and not response:
            continue

        blocks.append(
            f"""
Evidence {i}

Historical customer:
{customer}

Historical AmazonHelp response:
{response}
"""
        )

    return "\n".join(blocks)


def generate_with_openai(
    query: str,
    evidence: Iterable[dict],
    model: str | None = None,
) -> GeneratedResponse:

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI SDK is missing. Run: pip install openai"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    model = model or os.getenv("OPENAI_MODEL")

    if not model:
        raise RuntimeError(
            "Set OPENAI_MODEL to the API model you want to use."
        )

    client = OpenAI(api_key=api_key)

    user_prompt = f"""
Customer query:
{query}

Historical AmazonHelp evidence:
{build_evidence(evidence)}
"""

    result = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": user_prompt.strip(),
            },
        ],
    )

    raw = result.choices[0].message.content
    payload = json.loads(raw)

    return GeneratedResponse(
        draft_response=str(
            payload.get("draft_response", "")
        ).strip(),
        grounded_claim=str(
            payload.get("grounded_claim", "")
        ).strip(),
        needs_more_information=bool(
            payload.get("needs_more_information", False)
        ),
    )

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import AmazonHelpAgent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full AmazonHelp support agent."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Customer support message",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of historical interactions to retrieve",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Generation model. Defaults to OPENAI_MODEL.",
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

    print("\nCUSTOMER MESSAGE")
    print(result["query"])

    print("\nINTENT")
    print(json.dumps(
        result["intent"],
        indent=2,
        ensure_ascii=False,
    ))

    print("\nEVIDENCE ASSESSMENT")
    print(json.dumps(
        result["evidence_assessment"],
        indent=2,
    ))

    print("\nTOP HISTORICAL EVIDENCE")

    for item in result["retrieved_evidence"]:
        print(
            f"\nRank {item['rank']} "
            f"| dense_score={item['dense_score']:.4f}"
        )

        print(
            "Customer:",
            item["historical_customer_text"][:500],
        )

        if item.get("brand_response"):
            print(
                "AmazonHelp:",
                item["brand_response"][:500],
            )

    print("\nDRAFT RESPONSE")
    print(result["draft_response"])

    print("\nGROUNDED CLAIM")
    print(result["grounded_claim"])

    print(
        "\nNEEDS MORE INFORMATION:",
        result["needs_more_information"],
    )

    print(
        "\nSHOULD ESCALATE:",
        result["should_escalate"],
    )

    print("\nESCALATION REASONS")

    reasons = result["escalation_reasons"]

    if reasons:
        for reason in reasons:
            print("-", reason)
    else:
        print("none")


if __name__ == "__main__":
    main()

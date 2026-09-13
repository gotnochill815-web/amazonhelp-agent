from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generate import generate_with_openai


def load_evidence(golden_id: int) -> list[dict]:
    path = ROOT / "results" / "dense_comparison_top5.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing retrieval artifact: {path}"
        )

    df = pd.read_csv(path)

    rows = df[
        df["golden_example_id"].astype(int) == int(golden_id)
    ].sort_values("rank")

    if rows.empty:
        raise ValueError(
            f"No retrieval evidence found for golden_example_id={golden_id}"
        )

    return rows.to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a grounded AmazonHelp support reply."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Customer message",
    )

    parser.add_argument(
        "--golden-id",
        type=int,
        required=True,
        help="Golden Set example whose precomputed evidence should be used",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Optional API model. Defaults to OPENAI_MODEL.",
    )

    args = parser.parse_args()

    evidence = load_evidence(args.golden_id)

    result = generate_with_openai(
        query=args.query,
        evidence=evidence,
        model=args.model,
    )

    print("\n" + "=" * 70)
    print("AMAZONHELP SUPPORT AGENT")
    print("=" * 70)

    print("\nCUSTOMER MESSAGE")
    print(args.query)

    print("\nDRAFT RESPONSE")
    print(result.draft_response)

    print("\nGROUNDED CLAIM")
    print(result.grounded_claim)

    print("\nNEEDS MORE INFORMATION")
    print(result.needs_more_information)

    print("\nEVIDENCE USED")
    for i, item in enumerate(evidence, start=1):
        score = item.get("dense_score", "")
        print(
            f"{i}. rank={item.get('rank')} "
            f"score={score}"
        )


if __name__ == "__main__":
    main()

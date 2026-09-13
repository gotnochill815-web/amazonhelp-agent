
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

from .load import load_dataset
from .conversations import (
    build_customer_brand_pairs,
    build_parent_map,
    find_conversation_root,
)


BRAND = "AmazonHelp"


def build_amazonhelp_dataset(
    data_path: str,
    output_dir: str = "results",
) -> None:
    """Build clean AmazonHelp customer-support interaction data."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(data_path)

    print(f"Raw dataset: {len(df):,} tweets")

    print("\nBuilding customer -> brand response pairs...")
    pairs = build_customer_brand_pairs(df, BRAND)

    print(f"Direct customer -> AmazonHelp pairs: {len(pairs):,}")

    print("\nBuilding parent map...")
    parent_map = build_parent_map(df)

    interaction_ids = pd.concat(
        [
            pairs["customer_tweet_id"],
            pairs["brand_tweet_id"],
        ]
    ).drop_duplicates()

    print(f"Unique tweets involved: {len(interaction_ids):,}")

    print("\nReconstructing conversation roots...")

    roots = []

    for tweet_id in tqdm(
        interaction_ids,
        desc="Finding conversation roots",
    ):
        roots.append(
            find_conversation_root(
                tweet_id,
                parent_map,
            )
        )

    interactions = pd.DataFrame(
        {
            "tweet_id": interaction_ids.to_numpy(),
            "conversation_root": roots,
        }
    )

    conversation_sizes = (
        interactions
        .groupby("conversation_root")
        .size()
    )

    print("\nConversation statistics")
    print("=" * 50)
    print(f"Unique conversation roots: {len(conversation_sizes):,}")
    print(f"Mean relevant turns: {conversation_sizes.mean():.2f}")
    print(f"Median relevant turns: {conversation_sizes.median():.0f}")
    print(f"Maximum relevant turns: {conversation_sizes.max():,}")

    print("\nConversation depth")
    print("=" * 50)

    for threshold in [2, 3, 4, 5]:
        count = int((conversation_sizes >= threshold).sum())
        percentage = count / len(conversation_sizes) * 100

        print(
            f"{threshold}+ relevant turns: "
            f"{count:,} ({percentage:.1f}%)"
        )

    pairs_path = output_dir / "amazonhelp_customer_brand_pairs.csv"
    roots_path = output_dir / "amazonhelp_interaction_roots.csv"
    stats_path = output_dir / "amazonhelp_conversation_stats.csv"

    pairs.to_csv(pairs_path, index=False)

    interactions.to_csv(roots_path, index=False)

    stats = pd.DataFrame(
        [
            {
                "brand": BRAND,
                "amazonhelp_tweets": len(
                    df[
                        df["author_id"]
                        .astype(str)
                        .str.lower()
                        .eq(BRAND.lower())
                    ]
                ),
                "direct_customer_brand_pairs": len(pairs),
                "unique_customers": pairs[
                    "customer_author_id"
                ].nunique(),
                "unique_conversation_roots": len(
                    conversation_sizes
                ),
                "mean_relevant_turns": conversation_sizes.mean(),
                "median_relevant_turns": conversation_sizes.median(),
                "max_relevant_turns": conversation_sizes.max(),
            }
        ]
    )

    stats.to_csv(stats_path, index=False)

    print("\nSaved artifacts:")
    print(f"  {pairs_path}")
    print(f"  {roots_path}")
    print(f"  {stats_path}")


if __name__ == "__main__":
    # Update this path if the raw dataset is stored elsewhere.
    DATA_PATH = "/content/twcs/twcs/twcs.csv"

    build_amazonhelp_dataset(
        data_path=DATA_PATH,
        output_dir=str(Path(__file__).resolve().parents[2] / "results"),
    )

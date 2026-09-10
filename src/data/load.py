from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the Customer Support on Twitter dataset."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    return df

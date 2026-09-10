
from typing import Iterable

import numpy as np
import pandas as pd


def embed_texts(
    texts: Iterable[str],
    model,
    batch_size: int = 64,
) -> np.ndarray:
    """Generate normalized sentence embeddings."""

    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return np.asarray(embeddings)


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 15,
    random_state: int = 42,
):
    """Cluster embeddings using KMeans."""

    from sklearn.cluster import KMeans

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )

    labels = model.fit_predict(embeddings)

    return model, labels


def add_cluster_labels(
    df: pd.DataFrame,
    labels: np.ndarray,
    column: str = "cluster",
) -> pd.DataFrame:
    """Attach clustering labels to a dataframe."""

    if len(df) != len(labels):
        raise ValueError(
            "Number of labels must match dataframe length."
        )

    result = df.copy()
    result[column] = labels

    return result


def cluster_examples(
    df: pd.DataFrame,
    cluster_column: str = "cluster",
    text_column: str = "clean_text",
    examples_per_cluster: int = 8,
) -> dict[int, list[str]]:
    """Return representative examples for each cluster."""

    result = {}

    for cluster_id, group in df.groupby(cluster_column):
        examples = (
            group[text_column]
            .dropna()
            .head(examples_per_cluster)
            .tolist()
        )

        result[int(cluster_id)] = examples

    return result


from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

INPUT = RESULTS / "golden_set_human_review_completed.csv"

PREDICTIONS_OUT = RESULTS / "tfidf_human_gold_predictions.csv"
PER_INTENT_OUT = RESULTS / "tfidf_human_gold_per_intent.csv"
SUMMARY_OUT = RESULTS / "tfidf_human_gold_summary.csv"
CONFUSION_OUT = RESULTS / "tfidf_human_gold_confusion_matrix.csv"


def main():
    df = pd.read_csv(INPUT)

    required = [
        "golden_example_id",
        "customer_message",
        "human_intent",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()

    df["customer_message"] = (
        df["customer_message"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["human_intent"] = (
        df["human_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["customer_message"].ne("")
        & df["human_intent"].ne("")
    ].copy()

    print("=" * 80)
    print("TF-IDF HUMAN-GOLD BASELINE")
    print("=" * 80)
    print("Examples:", len(df))

    # ------------------------------------------------------------
    # Fixed holdout.
    #
    # We cannot stratify because several intents have only one
    # human-labelled example. Therefore we use a deterministic
    # 80/20 random holdout and explicitly document this limitation.
    # ------------------------------------------------------------

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    print("Train:", len(train))
    print("Test :", len(test))

    # ------------------------------------------------------------
    # TF-IDF
    # ------------------------------------------------------------

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_features=50000,
    )

    X_train = vectorizer.fit_transform(
        train["customer_message"]
    )

    X_test = vectorizer.transform(
        test["customer_message"]
    )

    # ------------------------------------------------------------
    # Multiclass logistic regression
    # ------------------------------------------------------------

    clf = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    clf.fit(
        X_train,
        train["human_intent"]
    )

    pred = clf.predict(X_test)

    y_true = test["human_intent"]

    accuracy = accuracy_score(y_true, pred)
    macro_f1 = f1_score(
        y_true,
        pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        pred,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)

    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    # ------------------------------------------------------------
    # Per-intent report
    # ------------------------------------------------------------

    labels = sorted(
        set(y_true) | set(pred)
    )

    report = classification_report(
        y_true,
        pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    per_intent = []

    for label in labels:
        r = report.get(label, {})

        per_intent.append({
            "intent": label,
            "precision": r.get("precision", 0.0),
            "recall": r.get("recall", 0.0),
            "f1": r.get("f1-score", 0.0),
            "support": r.get("support", 0),
        })

    per_intent_df = pd.DataFrame(per_intent)

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------

    predictions = test[
        [
            "golden_example_id",
            "customer_message",
            "human_intent",
        ]
    ].copy()

    predictions["predicted_intent"] = pred
    predictions["correct"] = (
        predictions["human_intent"]
        ==
        predictions["predicted_intent"]
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary = pd.DataFrame([
        {
            "dataset": "152-example human-reviewed Gold",
            "n_examples": len(df),
            "train_size": len(train),
            "test_size": len(test),
            "random_state": 42,
            "test_size_fraction": 0.20,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "evaluation_note": (
                "Fixed holdout used because singleton intents "
                "prevent valid stratified k-fold evaluation."
            ),
        }
    ])

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    predictions.to_csv(
        PREDICTIONS_OUT,
        index=False,
    )

    per_intent_df.to_csv(
        PER_INTENT_OUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUT,
        index=False,
    )

    cm_df.to_csv(
        CONFUSION_OUT
    )

    print("\nSaved:")
    print(PREDICTIONS_OUT)
    print(PER_INTENT_OUT)
    print(SUMMARY_OUT)
    print(CONFUSION_OUT)

    print("\n" + "=" * 80)
    print("PER-INTENT METRICS")
    print("=" * 80)

    print(
        per_intent_df.to_string(index=False)
    )


if __name__ == "__main__":
    main()

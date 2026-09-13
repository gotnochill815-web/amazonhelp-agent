
from pathlib import Path
import gc
import json
import time

import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


ROOT = Path(__file__).resolve().parents[2]

BM25_PATH = (
    ROOT / "results" / "bm25_comparison_top5.csv"
)

DENSE_PATH = (
    ROOT / "results" / "dense_comparison_top5.csv"
)

OUTPUT_PATH = (
    ROOT / "results"
    / "retrieval_comparison_judge.jsonl"
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


SYSTEM_PROMPT = """
You are evaluating historical evidence for an Amazon customer-support AI.

The CURRENT customer has a problem.

You are given FIVE historical customer-support interactions retrieved by
one retrieval method. Each interaction contains:
- historical customer message
- AmazonHelp response

Judge whether each historical interaction is useful evidence for answering
the CURRENT customer.

A historical interaction can be useful when it:
- addresses the same underlying issue
- addresses a closely related circumstance
- demonstrates a similar troubleshooting process
- demonstrates a similar support workflow
- demonstrates a similar resolution path

The historical response does NOT need to completely solve the current query.

Score usefulness:

2 = clearly useful evidence for answering the current customer
1 = somewhat useful / partially related
0 = not useful

Score actionability:

2 = contains a concrete action, workflow, troubleshooting step,
    or resolution that could inform the answer
1 = provides some useful context
0 = provides no useful guidance

Do not reward keyword overlap alone.

Return ONLY valid JSON:

{
  "items": [
    {
      "rank": 1,
      "usefulness": 0,
      "actionability": 0,
      "reason": "brief explanation"
    }
  ]
}

Return exactly five items, with ranks 1, 2, 3, 4, 5.
"""


def load_model():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading 7B model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model.eval()

    print("Model loaded.")

    return tokenizer, model


def extract_json(text):

    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            f"No JSON object found:\n{text}"
        )

    return json.loads(
        text[start:end + 1]
    )


def judge_method(
    query,
    candidates,
    tokenizer,
    model,
):

    evidence_block = ""

    candidates = candidates.sort_values(
        "rank",
        key=lambda s: s.astype(int)
    )

    for _, row in candidates.iterrows():

        evidence_block += f"""

EVIDENCE RANK {row["rank"]}

Historical customer:
{row["historical_customer_text"]}

Historical AmazonHelp response:
{row["brand_response"]}
"""

    prompt = f"""
CURRENT CUSTOMER:
{query}

{evidence_block}

Evaluate all five evidence items.

Return exactly five items.
Ranks must be 1, 2, 3, 4, 5.
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=700,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][
        inputs["input_ids"].shape[1]:
    ]

    raw = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    parsed = extract_json(raw)

    items = parsed.get("items", [])

    if len(items) != 5:
        raise ValueError(
            f"Expected 5 items, got {len(items)}"
        )

    cleaned = []

    for expected_rank, item in enumerate(
        items,
        start=1
    ):

        usefulness = int(
            item["usefulness"]
        )

        actionability = int(
            item["actionability"]
        )

        if usefulness not in [0, 1, 2]:
            raise ValueError(
                f"Invalid usefulness: {usefulness}"
            )

        if actionability not in [0, 1, 2]:
            raise ValueError(
                f"Invalid actionability: {actionability}"
            )

        cleaned.append(
            {
                "rank": expected_rank,
                "usefulness": usefulness,
                "actionability": actionability,
                "reason": str(
                    item.get("reason", "")
                ),
            }
        )

    return cleaned


def load_existing():

    completed = {}

    if not OUTPUT_PATH.exists():
        return completed

    with open(
        OUTPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            try:

                obj = json.loads(line)

                if "items" not in obj:
                    continue

                key = (
                    str(obj["method"]),
                    str(obj["golden_example_id"])
                )

                completed[key] = obj

            except Exception:
                continue

    return completed


def main():

    print("=" * 80)
    print("7B RETRIEVAL COMPARISON JUDGE")
    print("=" * 80)

    bm25 = pd.read_csv(
        BM25_PATH,
        dtype=str,
        keep_default_na=False
    )

    dense = pd.read_csv(
        DENSE_PATH,
        dtype=str,
        keep_default_na=False
    )

    print(
        f"BM25 rows: {len(bm25):,}"
    )

    print(
        f"Dense rows: {len(dense):,}"
    )

    # --------------------------------------------------------
    # Load model INSIDE script
    # --------------------------------------------------------
    tokenizer, model = load_model()

    completed = load_existing()

    print(
        f"Existing completed judgments: "
        f"{len(completed):,}"
    )

    methods = {
        "bm25": bm25,
        "dense": dense,
    }

    total = (
        bm25["golden_example_id"].nunique()
        + dense["golden_example_id"].nunique()
    )

    done = len(completed)

    with open(
        OUTPUT_PATH,
        "a",
        encoding="utf-8"
    ) as out:

        for method, df in methods.items():

            print()
            print(
                "=" * 60
            )
            print(
                f"METHOD: {method.upper()}"
            )
            print(
                "=" * 60
            )

            for example_id, group in df.groupby(
                "golden_example_id"
            ):

                key = (
                    method,
                    str(example_id)
                )

                if key in completed:
                    continue

                query = group["query"].iloc[0]

                try:

                    items = judge_method(
                        query,
                        group,
                        tokenizer,
                        model,
                    )

                    obj = {
                        "method": method,
                        "golden_example_id": str(
                            example_id
                        ),
                        "items": items,
                    }

                    out.write(
                        json.dumps(
                            obj,
                            ensure_ascii=False
                        ) + "\n"
                    )

                    out.flush()

                    completed[key] = obj
                    done += 1

                    if done % 10 == 0:

                        print(
                            f"{done}/{total} "
                            f"({done / total:.1%})"
                        )

                except Exception as e:

                    print(
                        f"ERROR "
                        f"{method} "
                        f"{example_id}: {e}"
                    )

                    # IMPORTANT:
                    # Do not write failed records into the
                    # completed checkpoint.
                    # They can therefore be retried.

    print()
    print("=" * 80)
    print("COMPARISON JUDGE COMPLETE")
    print("=" * 80)

    print(
        f"Completed: {len(completed):,}/{total:,}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    # Free GPU
    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

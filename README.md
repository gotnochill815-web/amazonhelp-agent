# AmazonHelp Support Agent

An evidence-aware AI support agent built for the Hiver SDE Intern take-home assignment using the Customer Support on Twitter dataset.

## Project Goal

Build a support agent that:

1. Classifies incoming customer messages into a small support-intent taxonomy.
2. Retrieves historically similar AmazonHelp interactions.
3. Grounds responses in how AmazonHelp handled similar issues.
4. Decides whether a case can be handled automatically or should involve a human.

The project emphasizes evaluation, leakage control, baselines, and explicit decision-making over model complexity.

## Dataset

Source: Customer Support on Twitter (TWCS)

Selected brand: **AmazonHelp**

The AmazonHelp corpus provides substantial conversational depth and diverse support issues including delivery, returns, refunds, payments, account issues, membership, and digital products.

The raw dataset is intentionally not committed to this repository.

## Intent Taxonomy

The current taxonomy contains 11 support intents:

- `delivery_status`
- `delivery_problem`
- `order_issue`
- `preorder_issue`
- `returns_replacements`
- `refund_issue`
- `payment_billing`
- `account_security`
- `prime_membership`
- `digital_product_support`
- `general_support`

Annotation rule: assign the primary actionable support request. Escalation is treated as a separate decision from intent classification.

## Data Split and Leakage Control

The historical retrieval corpus uses a temporal split so future evaluation conversations cannot leak into the historical evidence set.

The final retrieval corpus also excludes the six Golden Set conversation roots.

Validation includes:

- conversation-root overlap checks
- normalized exact-text overlap checks
- temporal separation between retrieval and evaluation data

## Retrieval Experiments

Three retrieval strategies were evaluated:

1. BM25
2. Dense retrieval using `all-MiniLM-L6-v2`
3. Hybrid BM25 + Dense retrieval followed by CrossEncoder reranking

Retrieval quality was evaluated using a local Qwen 2.5 7B judge with two dimensions:

- usefulness
- actionability

The LLM judge is treated as an evaluation instrument, not absolute ground truth.

The final retrieval component selected for the agent is **Dense retrieval** because it provided the strongest balance of useful and actionable historical evidence among the evaluated approaches.

## Dense Retrieval Calibration

Dense cosine similarity was separately calibrated against judged top-1 evidence quality on 200 Golden examples.

The experiment showed that higher similarity generally increased judged usefulness, but similarity alone was not a reliable signal for safe automatic resolution.

| Threshold | Coverage | Useful | Actionable |
|---:|---:|---:|---:|
| 0.70 | 76.5% | 71.9% | 2.6% |
| 0.75 | 55.5% | 75.7% | 2.7% |
| 0.80 | 34.5% | 81.2% | 1.4% |
| 0.85 | 14.5% | 72.4% | 3.4% |

Therefore the current implementation treats `0.70` as a **retrieval-quality gate**, not as a probability or automatic-answer confidence score.

## Agent Architecture

```text
Customer message
       |
       v
Intent classification
       |
       v
Dense top-k retrieval
       |
       v
Evidence assessment
       |
       +--------------------+
       |                    |
    sufficient           insufficient
       |                    |
       v                    v
Grounded draft       Clarification /
response             human escalation
       |
       v
Escalation / risk check
       |
       v
Final support decision
```

## Evaluation

The repository includes:

- human-labelled Golden Set
- TF-IDF intent baseline
- BM25 / Dense / Hybrid retrieval comparison
- LLM-as-judge retrieval evaluation
- Dense threshold calibration
- decision log documenting major modeling choices

The primary intent benchmark uses genuinely human-labelled examples. Model-assisted examples are kept separate from ground truth.

## Repository Structure

```text
src/
├── data/
├── intent/
├── evaluation/
├── retrieval/
└── agent.py

results/
reports/
notebooks/
```

## Author

**Prakhya Khandelwal**

GitHub: [@gotnochill815-web](https://github.com/gotnochill815-web)

## Current Status

- dataset reconstruction completed
- AmazonHelp corpus reconstructed
- temporal evaluation split completed
- Golden Set established
- human-labelled benchmark established
- TF-IDF intent baseline completed
- BM25 / Dense / Hybrid retrieval comparison completed
- Dense retrieval selected
- Dense evidence threshold calibration completed on 200 examples
- evidence-aware agent implementation in progress

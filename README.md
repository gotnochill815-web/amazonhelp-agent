# AmazonHelp AI Support Agent

AI customer-support agent built for the Hiver SDE Intern take-home assignment.

## Pipeline

Customer message -> intent classification -> dense historical retrieval -> evidence/actionability assessment -> grounded response or human escalation

## Final human-Gold evaluation

Primary benchmark: **152 genuinely human-labeled examples**.

| Metric | Result |
|---|---:|
| Intent accuracy | 56.6% |
| Intent macro F1 | 58.3% |
| Intent weighted F1 | 56.3% |
| Grounded rate | 98.0% |
| Strong evidence | 36.8% |
| High actionability | 42.1% |
| Auto-answer rate | 3.9% |

## Intent taxonomy

- delivery_status
- delivery_problem
- order_issue
- preorder_issue
- returns_replacements
- refund_issue
- payment_billing
- account_security
- prime_membership
- digital_product_support
- general_support

## Evaluation

The evaluation harness includes trivial and TF-IDF intent baselines, semantic intent experiments, BM25 retrieval, dense retrieval, hybrid retrieval, LLM-as-judge evaluation, grounding evaluation, and end-to-end human-Gold evaluation.

## Important caveat

A 98.0% grounded rate does not mean 98.0% of requests are safe to answer automatically.

Only 36.8% had strong evidence, 42.1% had high actionability, and the final auto-answer rate was 3.9%.

The system intentionally separates related retrieval from safe autonomous answering.

## Reports

- reports/FINAL_REPORT.md
- reports/FINAL_REPORT.pdf

## Decision log

See DECISION_LOG.md.

## Repository

https://github.com/gotnochill815-web/hiver-amazonhelp-agent
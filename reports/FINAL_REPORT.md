# AmazonHelp AI Support Agent
## Hiver SDE Intern Take-Home Assignment

## 1. Problem

Build an AI support agent that classifies incoming customer messages, retrieves historical AmazonHelp resolutions, drafts grounded replies, and decides whether to auto-answer or escalate.

The system prioritizes safe evidence use and evaluation over unnecessary model complexity.

## 2. Dataset and evaluation

AmazonHelp was selected from the Twitter Customer Support corpus because its support history contains diverse customer issues and multi-turn interactions.

The final primary human benchmark contains **152 genuinely human-labeled examples**.

Retrieval evaluation uses a temporal split and excludes Golden conversation roots from the retrieval corpus.

## 3. Intent taxonomy

The final taxonomy contains 11 support intents:

delivery_status, delivery_problem, order_issue, preorder_issue, returns_replacements, refund_issue, payment_billing, account_security, prime_membership, digital_product_support, general_support.

The final classifier combines high-precision boundary rules with a MiniLM semantic fallback.

## 4. Intent evaluation

| Metric | Result |
|---|---:|
| Accuracy | 56.6% |
| Macro F1 | 58.3% |
| Weighted F1 | 56.3% |

The human Gold set is treated as a fixed benchmark rather than repeatedly tuning individual examples.

## 5. Retrieval

BM25, dense MiniLM retrieval, and hybrid retrieval were evaluated. Dense retrieval was selected because it performed better on stricter usefulness and actionability measures than the broader hybrid result.

Similarity scores are treated as ranking signals, not calibrated probabilities.

## 6. Grounding and response policy

Finding a related historical conversation is not by itself sufficient for automatic answering.

The final policy requires strong evidence and high actionability before an answer is considered safe to automate.

The response generator is grounded in historical AmazonHelp responses and is instructed not to invent unsupported policies, refunds, eligibility conditions, or commitments.

## 7. End-to-end results

| Metric | Result |
|---|---:|
| Intent accuracy | 56.6% |
| Intent macro F1 | 58.3% |
| Intent weighted F1 | 56.3% |
| Grounded rate | 98.0% |
| Strong evidence | 36.8% |
| High actionability | 42.1% |
| Auto-answer rate | 3.9% |

The system auto-answered 6 cases and escalated 146 cases.

## 8. What is misleading about my headline number?

The most misleading headline number is the **98.0% grounded rate**.

That number does not mean the agent can safely answer that percentage of requests.

Only **36.8%** of cases had strong evidence and **42.1%** had high actionability. The resulting automatic-answer rate was only **3.9%**.

Therefore:

**related retrieval != strong evidence != actionable evidence != safe auto-answer**

This distinction is intentional and forms the main safety property of the system.

## 9. Human evaluation and limitations

Intent agreement between the system and the human Gold labels was 56.6%.

Escalation labels are highly imbalanced in the human benchmark, so escalation agreement is treated as diagnostic rather than a definitive production safety estimate.

The LLM judge is treated as an evaluation instrument rather than ground truth. A larger independent human review set would be appropriate for production validation.

## 10. Engineering decisions

The final system prioritizes:

1. leakage-safe temporal evaluation
2. Golden-root exclusion from retrieval
3. dense retrieval selected using stricter quality criteria
4. explicit intent boundaries
5. conservative evidence and actionability gating
6. escalation instead of unsupported generated claims

## Repository

https://github.com/gotnochill815-web/hiver-amazonhelp-agent
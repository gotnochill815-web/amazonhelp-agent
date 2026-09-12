# Decision Log

## Brand selection
Selected AmazonHelp because its corpus provides substantial support history across diverse customer issues.

## Evaluation split
Used a temporal split rather than a random retrieval split to reduce future-information leakage.

## Golden-set leakage protection
Golden conversation roots were excluded from the historical retrieval corpus.

## Intent taxonomy
Used 11 compact actionable support intents. Escalation is treated separately from intent.

## Intent classifier
Selected a hybrid of narrow high-precision rules and MiniLM semantic fallback after comparing several alternatives.

## Retrieval
Evaluated BM25, dense, and hybrid retrieval. Dense retrieval was selected based on stronger usefulness and actionability criteria.

## Similarity threshold
Dense similarity is treated as a ranking signal rather than a calibrated probability.

## Grounding
Automatic response requires strong evidence and high actionability.

## Response generation
Historical AmazonHelp responses are used as grounding. The generator should not invent unsupported policies or commitments.

## Escalation
The system escalates when evidence is insufficient, actionability is weak, intent confidence is uncertain, additional information is required, or the case is sensitive.

## Final benchmark
Human-Gold examples: 152
Intent accuracy: 56.6%
Intent macro F1: 58.3%
Grounded rate: 98.0%
Strong evidence: 36.8%
High actionability: 42.1%
Auto-answer rate: 3.9%

## Main lesson
related retrieval != strong evidence != actionable evidence != safe auto-answer
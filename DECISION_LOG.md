# Decision Log

## Retrieval architecture selection

### Decision

Use dense retrieval as the final retrieval component for the support agent.

### Evidence

Evaluation used 200 Golden queries and five retrieved candidates per query
for BM25, dense, and hybrid retrieval.

At K=5:

- BM25: 86.5% useful, 69.0% clearly useful, 75.0% actionable
- Dense: 87.0% useful, 76.0% clearly useful, 81.0% actionable
- Hybrid + CrossEncoder: 90.0% useful, 23.0% clearly useful, 29.0% actionable

### Rationale

Hybrid achieves the highest broad usefulness rate, but that metric counts
partially useful evidence. Dense retrieval provides substantially more
clearly useful and actionable evidence, which is more important for grounded
support responses.

### Rejected alternative

Hybrid + CrossEncoder was rejected for the final agent despite its higher
Useful@5 score because its evidence quality was materially lower.

### Evaluation caveat

The retrieval judge is an LLM-based evaluation instrument. The metrics should
be interpreted as judged evidence usefulness rather than absolute retrieval
ground truth.

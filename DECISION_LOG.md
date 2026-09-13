# Decision Log

## 1. Selected AmazonHelp as the target brand
**Decision:** Use AmazonHelp rather than simply choosing the largest support account.

**Why:** AmazonHelp provides broad coverage across delivery, orders, refunds, returns, payments, accounts, Prime, and digital products. This creates enough variety for intent classification, retrieval, grounding, and escalation to be meaningful.

**Trade-off:** A broader issue mix increases taxonomy ambiguity and makes the classifier harder than a single-issue support dataset.

---

## 2. Defined a compact 11-intent taxonomy
**Decision:** Use 11 actionable support intents rather than a large unconstrained taxonomy.

**Why:** The goal is an operational routing system, so intents need to correspond to meaningful support workflows.

**Trade-off:** Some real conversations contain multiple issues and must be mapped to one primary requested resolution.

---

## 3. Separated intent from escalation
**Decision:** Model the customer's intent independently from whether the request should be automated.

**Why:** A correct intent does not imply that the request is safe to answer automatically. Account security and payment issues, for example, can have clear intent but still require human handling.

**Trade-off:** The system needs a second decision layer instead of treating classification confidence as automation confidence.

---

## 4. Used a temporal evaluation split
**Decision:** Keep evaluation conversations later in time than the retrieval corpus.

**Why:** Random retrieval splits can leak highly related or identical conversations across evaluation and retrieval.

**Trade-off:** The historical retrieval set is smaller than the full corpus and may be less convenient than a random split, but it gives a more defensible estimate of historical generalization.

---

## 5. Excluded Golden-set conversation roots from retrieval
**Decision:** Remove the conversation roots represented in the Gold set from the historical retrieval corpus.

**Why:** This prevents direct or near-direct retrieval leakage from the evaluation examples.

**Trade-off:** A small amount of potentially useful historical data is sacrificed for evaluation integrity.

---

## 6. Used human labels as primary ground truth
**Decision:** Treat the 152 genuinely human-labelled examples as the primary Gold set.

**Why:** Model-assisted labels can introduce confirmation bias and make evaluation circular.

**Trade-off:** The Gold set is relatively small, so rare intents remain difficult to evaluate robustly.

---

## 7. Included difficult and uncertain examples in the human review set
**Decision:** Prioritize difficult boundary cases rather than relying only on easy random examples for qualitative review.

**Why:** The most important failure modes occur near intent boundaries and automation boundaries.

**Trade-off:** This review sample is not representative of the full corpus, so its results are diagnostic rather than population estimates.

---

## 8. Rejected keyword-only topic discovery as the final taxonomy method
**Decision:** Use embedding-based semantic exploration to inform taxonomy design rather than relying on keyword counts alone.

**Why:** Keyword topics were brittle and left many customer messages unmatched, while embeddings exposed semantic clusters such as delivery, preorder, refund, account/security, and digital-product issues.

**Trade-off:** Embedding-based exploration is more computationally expensive and still requires human interpretation.

---

## 9. Selected rule + MiniLM intent classification
**Decision:** Combine high-precision boundary rules with MiniLM semantic fallback.

**Why:** Rules perform well on obvious boundary phrases while MiniLM provides coverage for less explicit language.

**Trade-off:** Rules require maintenance and can still conflict with semantic predictions.

---

## 10. Rejected pure LLM few-shot intent classification as the production classifier
**Decision:** Do not use the LLM alone as the final intent classifier.

**Why:** On the fixed human Gold holdout, the LLM-only approach did not outperform the final rule + MiniLM system.

**Trade-off:** The final classifier is less flexible than free-form LLM reasoning but is cheaper, more deterministic, and easier to evaluate.

---

## 11. Compared BM25, dense, and hybrid retrieval
**Decision:** Benchmark lexical, semantic, and hybrid retrieval instead of assuming hybrid retrieval is automatically best.

**Why:** Hybrid retrieval had the highest broad Useful@5 score but substantially weaker StrongUseful and Actionable results in the judge evaluation.

**Trade-off:** Dense retrieval sacrifices some lexical precision but produced stronger evidence quality for this task.

---

## 12. Selected dense retrieval for the final system
**Decision:** Use `all-MiniLM-L6-v2` dense retrieval.

**Why:** Dense retrieval achieved stronger strong-usefulness and actionability than the hybrid approach under the final evaluation rubric.

**Trade-off:** Dense retrieval can miss exact lexical matches and requires embedding computation.

---

## 13. Treated similarity as a pre-filter, not calibrated confidence
**Decision:** Use dense similarity to reject obviously weak evidence but do not interpret the similarity score as a probability of correctness.

**Why:** Threshold calibration showed that similarity correlated imperfectly with actionability and safe autonomous handling.

**Trade-off:** The system abstains more often than an aggressive similarity-based automation policy.

---

## 14. Required historical grounding for generation
**Decision:** Restrict the response generator to the customer message and retrieved historical AmazonHelp evidence.

**Why:** This reduces unsupported policy claims, invented refunds, eligibility statements, and commitments.

**Trade-off:** Responses can become cautious or generic when the historical evidence is incomplete.

---

## 15. Used conservative automation gating
**Decision:** Escalate when evidence is weak, intent is uncertain, required information is missing, or the issue is high-risk.

**Why:** A support system should optimize for safe handling, not maximum auto-answer rate.

**Trade-off:** The final auto-answer rate is only 3.9%, but this is preferable to confidently hallucinating resolutions.

---

## 16. Used human review as the primary qualitative evaluation
**Decision:** Treat human review as the primary qualitative signal and LLM-as-judge as a secondary evaluator.

**Why:** The 20-case human vs LLM comparison showed low agreement across reply-quality dimensions, including negative Cohen's kappa values on some dimensions.

**Trade-off:** Human review is slower and more expensive, but evaluator disagreement itself became a useful failure signal.

---

## 17. Kept the LLM-as-judge disagreement results
**Decision:** Preserve the disagreement artifacts instead of selecting only favorable examples.

**Why:** Disagreement reveals evaluator-calibration problems and prevents the evaluation from presenting an artificially clean story.

**Trade-off:** The resulting report contains weaker agreement numbers, but those numbers make the evaluation more honest.

---

## 18. Kept model-assisted Gold examples separate
**Decision:** Retain the additional 48 model-assisted examples as exploratory data only.

**Why:** They can be useful for future analysis without contaminating the primary human-labelled benchmark.

**Trade-off:** The headline evaluation uses fewer examples.

---

## 19. Chose reproducible precomputed artifacts
**Decision:** Commit evaluation artifacts and repository-relative code while excluding heavy raw data and model weights.

**Why:** Reviewers can inspect the reported results without rerunning every expensive experiment.

**Trade-off:** Full reproduction of the historical retrieval corpus still requires the original dataset to be downloaded separately.

---

## 20. Scoped the production claim deliberately
**Decision:** Present the project as a conservative support-triage and grounded-response prototype, not as a fully autonomous customer-service replacement.

**Why:** The evaluation shows a large gap between finding relevant evidence and having evidence strong enough for autonomous handling.

**Trade-off:** The headline automation claim is intentionally modest, but it is better aligned with the actual evidence.


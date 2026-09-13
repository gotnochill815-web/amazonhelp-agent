# AmazonHelp AI Support Agent

AI customer-support agent built for the Hiver SDE Intern take-home assignment,
using the `thoughtvector/customer-support-on-twitter` dataset, filtered to the
AmazonHelp brand.

## Live demo

**Streamlit app:** https://hiver-amazonapp-agent-mbpjvg7jbtk2vepeq3jhs4.streamlit.app/

The interactive deployment runs the real pipeline (`src/agent.py`) end to
end against a compact 500-row temporal-safe historical subset
(`data/demo_retrieval.csv`), for responsiveness. The evaluation numbers
below were produced on the full retrieval corpus, not this subset — treat
the app as a live illustration of the decision process, not as the source
of the headline metrics.

---

## Pipeline

```
customer message
   -> intent classification (rule-based + MiniLM centroid fallback)
   -> dense historical retrieval (MiniLM embeddings over past AmazonHelp resolutions)
   -> evidence-quality assessment (is the best match above a similarity gate?)
   -> conditional grounded response generation (only if evidence is sufficient)
   -> escalation decision, with stated reasons
```

The core design principle: **finding a related past conversation is not the
same thing as having enough evidence to safely auto-answer.** See
`reports/FINAL_REPORT.md`, section 6, for why.

---

## Quickstart (~2 minutes, no dataset download required)

The repo ships with precomputed artifacts (`results/golden_set_human_review_completed.csv`
for intent training, `results/dense_comparison_top5.csv` as a fallback retrieval
corpus), so you can run the full agent end-to-end without downloading the raw
Twitter dataset first.

```bash
git clone https://github.com/gotnochill815-web/hiver-amazonhelp-agent.git
cd hiver-amazonhelp-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"   # any chat-completions model with JSON mode

python -m src.agent --query "My package says delivered but I never got it, it's been 3 days"
```

Expected output: the classified intent, the top retrieved historical
interactions with similarity scores, a grounded draft reply (or an
escalation message if evidence is too weak), and the escalation decision
with reasons. `scripts/run_agent.py` runs the same pipeline via a CLI
wrapper with JSON-formatted intermediate output, if you want that instead.

To run the same thing locally as the Streamlit demo above:

```bash
streamlit run app.py
```

---

## Reproducing the headline evaluation numbers

The 152-example human-labelled Gold set and all retrieval/intent baseline
predictions are already committed under `results/`, so the headline table
below can be reproduced by reading those artifacts directly — no retraining
or re-embedding required:

- `results/end_to_end_human_gold_evaluation.csv` — per-example outcomes
- `results/end_to_end_metrics.csv` — the aggregated headline table
- `results/final_intent_classification_report.csv`, `final_intent_confusion_matrix.csv`
- `results/intent_baseline_metrics.csv`, `intent_classifier_comparison.csv` — baseline comparisons
- `results/dense_comparison_top5.csv`, `bm25_comparison_top5.csv`, `hybrid_retrieval_top5.csv` — retrieval comparisons

To regenerate intent predictions or retrieval rankings from scratch instead
of reading the committed CSVs, run the relevant script under `src/intent/`
or `src/retrieval/` directly — each has its own `if __name__ == "__main__"`
block. These re-embed with `sentence-transformers` locally (CPU is fine,
a few minutes for ~150-1000 rows).

**Not included in the 15-minute path:** `src/evaluation/judge_retrieval_comparison.py`
runs `Qwen/Qwen2.5-7B-Instruct` locally via `transformers` + `bitsandbytes` as an
LLM judge for retrieval-quality labelling. This needs a GPU and a multi-GB
model download, and its outputs are already committed as
`results/retrieval_comparison_judge.jsonl` / `results/llm_judge_retrieval_metrics.csv`.
You do not need to re-run it to reproduce the headline numbers.

---

## Optional: rebuilding from the raw dataset

If you want to rebuild the AmazonHelp corpus from scratch instead of using
the committed artifacts:

1. Download `thoughtvector/customer-support-on-twitter` from Kaggle and
   extract `twcs.csv`.
2. Edit the `DATA_PATH` at the bottom of `src/data/build_amazonhelp.py` to
   point at your local `twcs.csv`.
3. Run:
   ```bash
   python -m src.data.build_amazonhelp
   ```
   This filters to AmazonHelp threads, builds the temporal train/eval split,
   and excludes Golden-set conversation roots from the retrieval corpus
   (see `DECISION_LOG.md` for why). Outputs are written to `results/`.

This step is not required to reproduce the headline metrics above — it's
only needed if you want to regenerate the underlying corpus itself.

---

## Final human-Gold evaluation

Primary benchmark: **152 genuinely human-labeled examples**
(`results/golden_set_human_review_completed.csv`).

| Metric | Result |
|---|---:|
| Intent accuracy | 56.6% |
| Intent macro F1 | 58.3% |
| Intent weighted F1 | 56.3% |
| Grounded rate | 98.0% |
| Strong evidence | 36.8% |
| High actionability | 42.1% |
| Auto-answer rate | 3.9% |

See `reports/FINAL_REPORT.md` section 6 ("What is misleading about my
headline number?") before treating the 98.0% grounded rate as an accuracy
figure — it isn't one.

### vs. baselines (intent classification, human-Gold holdout)

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority-class baseline | 32.3% | 7.0% | 15.7% |
| TF-IDF + Logistic Regression | 25.8% | 18.9% | 24.8% |
| Final (rule + MiniLM) | 45.2% | 35.7% | 42.5% |

(Note: this table uses a separate fixed classifier holdout; the 56.6%
end-to-end figure above is measured on the full human-Gold set including
rule-based short-circuits — see `reports/FINAL_REPORT.md` for the distinction.)

---

## Intent taxonomy

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

Escalation is modeled separately from intent (see `src/agent.py`,
`decide_escalation`), not as an intent class.

---

## Repository structure

```
app.py                        # Streamlit demo, calls src/agent.py directly
data/demo_retrieval.csv       # 500-row retrieval subset used by the live demo
src/
  agent.py                    # end-to-end agent: intent -> retrieval -> gating -> generation -> escalation
  generate.py                  # OpenAI-backed grounded reply drafting
  data/build_amazonhelp.py     # raw twcs.csv -> AmazonHelp corpus + temporal split
  intent/                      # intent taxonomy, classifiers, baselines
  retrieval/                   # BM25, dense, hybrid retrieval experiments
  evaluation/                  # golden-set sampling, LLM-judge, baseline eval scripts
scripts/
  run_agent.py                  # CLI wrapper around the same end-to-end pipeline
results/                        # all precomputed predictions, metrics, and golden-set data
reports/
  FINAL_REPORT.md / .pdf        # full write-up: framing, baselines, failure analysis, next steps
DECISION_LOG.md                 # non-obvious decisions and rationale
```

## Evaluation harness

Includes: trivial and TF-IDF intent baselines, MiniLM centroid/kNN intent
experiments, BM25 retrieval, dense retrieval, hybrid retrieval, LLM-as-judge
retrieval evaluation, grounding evaluation, and end-to-end human-Gold
evaluation. See `reports/FINAL_REPORT.md` for the full methodology and
`DECISION_LOG.md` for why each was chosen over the alternatives.

## Reports

- `reports/FINAL_REPORT.md`
- `reports/FINAL_REPORT.pdf`

## Decision log

See `DECISION_LOG.md`.

## Repository

https://github.com/gotnochill815-web/hiver-amazonhelp-agent

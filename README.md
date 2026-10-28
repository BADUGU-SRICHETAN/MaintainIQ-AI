# MaintenIQ AI

Hybrid retrieval and cited investigation for industrial maintenance records. A question is answered
only from retrieved evidence. Every claim carries a citation, which is validated. The system refuses
when the records do not support an answer.

> **Data notice.** All manuals, work orders, and sensor readings are synthetic. They were generated
> for development and evaluation. Nothing here is real plant data.

## What it does

- **Hybrid retrieval.** BM25 handles exact terms and fault codes such as `BRG-WEAR`. A dense
  TF-IDF + LSA index handles paraphrases. Reciprocal Rank Fusion combines the two rankings, so their
  score scales never need to match.
- **Metadata filtering.** Filter by equipment. Shared manuals remain visible to every motor.
- **Cited investigation.** The answer cites evidence as `[E1]`, `[E2]`, and so on. A validator checks
  that each cited ID was actually retrieved, that each claim has a citation, and that each cited
  claim shares content terms with its evidence.
- **Refusal gates.** Before generation, the system refuses when the top evidence is not semantically
  close enough, when fewer than half of the question's content terms appear in the evidence, or when
  an exact-value question has no number in the evidence. It applies a second numeric check to the answer itself.
- **Sensor analytics.** Baseline z-scores, rolling anomaly detection, and linear trends per motor.
  Summaries are indexed as evidence, so a report can cite current sensor state alongside history.
- **Dashboard.** Streamlit. Investigation with cited evidence, a retrieval trace showing BM25 and dense
  ranks and RRF scores, and sensor analytics with charts.

## Architecture

```
manuals (.md) ──► section split ──► chunks ─────────┐
work orders (csv) ─► one record each ───────────────┤
sensor readings ─► per-motor summaries ─────────────┤
                                                    ▼
                         BM25 index ◄──────── corpus ────────► dense index (TF-IDF + LSA)
                                                    │
                                 Reciprocal Rank Fusion (k = 60, metadata filters)
                                                    │
                          refusal gates (similarity · term coverage · numeric)
                                                    │
                      generation: Ollama if available, else extractive fallback
                                                    │
                       citation validator ──► cited answer + validation metrics
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                               # or: pip install -r requirements-dev.txt
python scripts/generate_data.py                       # synthetic data + golden sets
streamlit run app/streamlit_app.py
```

Command-line checks:

```bash
python scripts/smoke_investigate.py "Why is the motor vibrating at double running speed?"
python scripts/evaluate.py          # retrieval metrics  -> reports/retrieval_eval.json
python scripts/evaluate_answers.py  # answer metrics     -> reports/answer_eval.json
pytest                              # 44 tests
ruff check src tests scripts app
```

To use a local LLM, install [Ollama](https://ollama.com), run `ollama pull llama3.2`, and copy
`.env.example` to `.env`. Without a model the system falls back to an extractive answer and reports
the backend it used.

## Results

Retrieval, on the synthetic golden set (`eval/golden_set.jsonl`). Queries are paraphrases that do not
reuse the symptom wording. Relevance labels come from construction: each query belongs to a fault
family and is relevant to its manual section and its work orders.

| Split | Queries | Precision@5 | MRR | Hit rate@5 | Recall@5 |
|---|---|---|---|---|---|
| dev  | 10 | 0.76 | 0.91 | 0.90 | 0.45 |
| test | 5  | 0.80 | 0.80 | 0.80 | 0.55 |

Recall@5 is low because each fault has about ten relevant work orders, and a top-5 list cannot cover
them. The report includes the ceiling, which is 0.63 for these splits. Precision and MRR are the
more useful numbers here.

Answers, on 11 labelled questions (`eval/answer_set.jsonl`):

| Metric | Value |
|---|---|
| Refusal accuracy | 1.00 (6 answerable answered, 5 unanswerable refused) |
| Citation IDs that do not exist | 0 |
| Cited claims with lexical support | 11 of 11 |

**This answer set is not held out.** I wrote these questions after seeing the failure cases, and I tuned
the refusal and extraction gates against them. Treat 1.00 as a regression check. It shows the fixes work,
not that the system generalizes. A real evaluation needs questions written before tuning, and
ideally a reviewer who has not seen the code.

## Design decisions

- **RRF over weighted score fusion.** BM25 scores and cosine similarities have different scales,
  and rank-based fusion avoids normalizing them.
- **Stemming and stopword removal in the shared tokenizer.** Without this, the dense index treated
  "the" and "of" as signal, and an unrelated query scored 0.82 similarity. After the change the same
  query scores 0.0.
- **Term-coverage gate.** Similarity alone cannot tell "pressure sensor cost" from "sensor summary"
  when the two share one word. Requiring query terms to appear in the evidence catches that.
- **Shared documents pass equipment filters.** Filtering on equipment removed the manuals entirely
  until this was fixed.
- **Retrieval is evaluated separately from generation.** A correct retriever with a poor generator
  and the reverse are different failures and need different fixes.
- **Recall is reported with its ceiling.** Plain recall@k can look poor for reasons unrelated to
  retrieval quality.

## Known limitations

- **Lexical citation checks are not semantic verification.** Word overlap can pass a wrong claim
  and fail a correct paraphrase. Human review is still required.
- **The extractive fallback cannot synthesize.** It quotes the most relevant sentence from each of the
  top documents. Answers are less fluent than with an LLM, but they cannot introduce unsupported content.
- **Missing specifics can still pass.** "What was the exact operating temperature of MTR-003 before the
  bearing was replaced?" can match a sensor summary that contains a temperature. The system does not know
  the question refers to an event that is not in the records. It is worth testing this case with real records.
- **Synthetic data only.** The fault catalogue and paraphrases were written by the same author as the
  code, so the golden set shares that author's assumptions.
- **Untested paths.** The Ollama path is tested with mocked HTTP only. No local model was run.
  The `sentence-transformers` backend is implemented but not tested, because the model download was
  unavailable. The Dockerfile is written but not built.

## Roadmap

- Run the Ollama path against a local model and record generated-answer metrics.
- Write a held-out answer set before any further tuning, and have someone outside the project review it.
- Add a cross-encoder reranker over the top candidates, and measure its effect against the current ranking.
- Add incident similarity: find past work orders with the same symptom pattern and outcome.
- Re-add the AI4I failure classifier as a separate module, with its own evaluation. It is not part
  of this rebuild.

## Repository layout

```
src/maintainiq/     package: ingestion, retrieval, citations, investigation, sensors, evaluation
app/                Streamlit dashboard
scripts/            data generation and evaluation entry points
eval/               golden retrieval set and answer set
tests/              pytest suite (data, ingestion, retrieval, citations, sensors, investigation, UI, LLM)
reports/            generated evaluation JSON
```

## License

MIT. See `LICENSE`.

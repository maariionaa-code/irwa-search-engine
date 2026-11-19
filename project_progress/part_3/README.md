# IRWA Project: Part 3
## Ranking and Filtering

Part 3 keeps the conjunctive (AND) search of Part 2 fixed and compares six ways of **ordering** its results. All methods rank exactly the same candidate set, so every difference between them comes from the scoring function alone.

---

## Ranking methods

| Method | File → function | Formula / idea |
|---|---|---|
| TF-IDF + cosine | `ranking_methods.py` → `rank_tfidf` | weight = (1 + ln tf) · (ln((N+1)/(df+1)) + 1); cosine(query, product). Identical to Part 2 (tested). |
| BM25 | `ranking_methods.py` → `rank_bm25` | Σ idf · tf(k1+1) / (tf + k1(1 − b + b·\|d\|/avgdl)), with idf = ln(1 + (N − df + 0.5)/(df + 0.5)), k1 = 1.2, b = 0.75 |
| Our score | `ranking_methods.py` → `rank_our_score` | stock × (0.55·T + 0.25·C + 0.15·R + 0.05·D) |
| Intent score | `intent_score.py` → `IntentScorer.rank` | T × K × G × A × S |
| word2vec + cosine | `embedded_ranking.py` → `rank_word2vec` | product vector = mean of its word vectors; cosine with the query vector |
| Doc2Vec + cosine | `embedded_ranking.py` → `rank_doc2vec` | learned document vectors (PV-DBOW); query vector inferred |

### Our score (relevance + product quality)

| Component | Definition | Weight |
|---|---|---|
| T, text | BM25 / best BM25 among the query's candidates | 0.55 |
| C, title coverage | share of the distinct query terms that appear in the title | 0.25 |
| R, rating | (rating − 1) / 4; a missing rating gets the corpus median (neutral) | 0.15 |
| D, discount | discount % / 100; missing = 0 | 0.05 |
| stock | 1 if in stock, 0.5 if out of stock (multiplies the whole score) | × |

### Intent score (relevance + what the user meant)

| Factor | Definition | Range |
|---|---|---|
| T, text | BM25 / best BM25 among the query's candidates | (0, 1] |
| K, category coherence | pseudo-relevance feedback: share of the product's sub-category among the top-20 BM25 results, relative to the most common one | [0.5, 1] |
| G, gender | query gender ("men", "women") vs the product's "Ideal For" field, or its title if that field is missing | match 1 · unknown 0.7 · mismatch 0.3; 1 for queries without gender |
| A, attributes | query words that are values of structured attributes (Fit, Fabric, Sleeve, Color, Generic Name, Type, …; yes/no fields such as Hooded) are checked against the product: agree / contradict / unknown | [0.5, 1] |
| S, stock | as in our score | 0.5 or 1 |

---

## Evaluation design

- **Queries:** Q1–Q5 from Part 2, plus Q6 "black track pants", Q7 "hooded sweatshirt" and Q8 "denim jacket". Q6–Q8 mention no gender; they were added after the Part 2 feedback.
- **Pooling:** for each query, the union of every method's top 20 is judged. Part 2 labels are reused automatically. `export_judgment_pool.py` writes the union of the existing pool and the current top-20 pairs, so a judged product that no method retrieves any more keeps its label (`in_current_top20` = False) and still counts in the denominators. Only `found_by` / `in_current_top20` are refreshed. A pool label that differs from Part 2 is printed until it is marked `label_source = part2_corrected` (one such case: Q5 `SRTFWYYDASDR42FF`, fabric "Hosiery" is a knit, the description says cotton).
- **Same denominators for all methods:** R is the number of judged relevant products in the pool, so recall, AP and NDCG are comparable across the six methods. They are *not* directly comparable with the Part 2 numbers, whose pool (one method's top 20) was smaller.
- **Judging rules:**
    - A product is relevant if it matches the product type, the audience (if stated) and every stated attribute.
    - When fields conflict, they are trusted in this order: title > structured attributes > description > category/URL.
    - A blend counts as the material ("cotton blend" satisfies "cotton").
- **Metrics:**
    - Computed at K = 5, 10, 15 and 20: P@K, R@K, F1@K, AP@K/MAP@K, NDCG@K and RR@K/MRR@K.
    - AP@K = Σ_{i≤K} P@i · rel_i / min(R, K) (truncated AP, as in Part 2). RR@K is truncated at K.
    - Unjudged products count as non-relevant. This makes P@K a lower bound, but not R, AP or NDCG (the true R is unknown).
    - Paired randomisation tests against BM25 and TF-IDF (exact over all 2^8 sign assignments). The 30 tests are exploratory; `significance.csv` reports both the raw p-values and Holm-adjusted ones over the whole family.
- **Ablation judgments:** products that only an ablation variant or a BM25 grid cell puts in its top 10 are judged in a separate file, `judgments_ablation.csv`, so the method comparison is unchanged. The ablation uses the pool plus these labels, so every variant is evaluated with judged@10 = 10; `--pool-only` reproduces the pool-only views (unjudged = non-relevant, and condensed). Re-exporting keeps every existing row, label and annotation column.
- **Label agreement (prepared, not yet run):** `label_agreement.py` draws a blind, stratified sample of 100 pool pairs (no proposed labels, no `found_by`, no URL) for a second group member, and computes observed agreement and Cohen's kappa.
- **Independent check:** every method is also evaluated on the professor's validation labels (`evaluate_validation_methods.py`).

---

## How to run (from the repository root)

```bash
pip install -r requirements.txt          # adds gensim and scipy
python -m nltk.downloader stopwords
python -m pytest project_progress/part_3 -v

python -m project_progress.part_3.query_term_stats
python -m project_progress.part_3.run_rankings              # trains word2vec/Doc2Vec on the first run
python -m project_progress.part_3.inspect_embeddings        # sanity check of the word2vec model
python -m project_progress.part_3.export_judgment_pool      # then label judgments_pool.csv (1/0)
python -m project_progress.part_3.evaluate_methods
python -m project_progress.part_3.evaluate_validation_methods
python -m project_progress.part_3.compare_rankings
python -m project_progress.part_3.ablation --export-unjudged   # then label judgments_ablation.csv
python -m project_progress.part_3.ablation
python -m project_progress.part_3.ablation --pool-only         # pool labels only -> results/*_pool_only.csv
python -m project_progress.part_3.label_agreement --export     # a second person labels agreement_sample.csv
python -m project_progress.part_3.label_agreement
python -m project_progress.part_3.explain_scores
python -m project_progress.part_3.make_figures
```

To search with any query and method:

```bash
python -m project_progress.part_3.pipeline "men running shoes" --method intent_score --top-k 10
```

---

## Outputs

| File | Content |
|---|---|
| `results/query_term_stats.csv` | df, share of corpus, TF-IDF and BM25 IDF of every query term |
| `results/candidate_counts.csv` | size of the AND result per query |
| `results/rankings_top20.csv` | top 20 per query and method, with score, rating, discount, stock and length |
| `judgments_pool.csv` | pooled relevance labels (`label_source` = part2 or part3, `found_by` = methods) |
| `results/method_metrics.csv`, `results/method_summary.csv` | metrics per method/query/K and their means |
| `results/query_group_summary.csv` | mean P@10 and MAP@10 for queries with a gender (Q1–Q5) and without (Q6–Q8) |
| `results/significance.csv` | paired randomisation tests (K = 10) against BM25 and TF-IDF |
| `results/validation_methods*.csv` | every method on the professor's validation labels |
| `results/method_agreement.csv` | overlap@5/10/20, Jaccard@20, RBO@20 and Kendall's tau per method pair |
| `results/score_spread.csv` | scores at ranks 1/10/20 and near-ties around rank 10 |
| `results/tfidf_vs_bm25_rank_shifts.csv` | products that move most between TF-IDF and BM25, with length and term frequencies |
| `judgments_ablation.csv` | labels for products that only the ablation variants / BM25 grid put in a top 10 (`surfaced_by` = which ones) |
| `agreement_sample.csv` | blind sample of 100 pool pairs for a second annotator |
| `results/ablation.csv`, `results/bm25_grid.csv` | ablations of both original scores; BM25 k1 × b grid; each with `judged@10` and judged-only ("condensed") P@10 / AP@10 |
| `results/score_components.csv` | every component of both original scores for each top-20 product |
| `results/word2vec_neighbours.csv` | nearest neighbours of every query term |
| `figures/*.png` | P@10 by query, MAP@K by K, Kendall's tau heatmap |

The trained models are stored in `models/`, which is git-ignored because they can be rebuilt with the same seed (`run_rankings --retrain`). gensim seeds its initial vectors with Python's `hash`, which changes in every new process, so `embedded_ranking.py` passes a stable hash (`stable_hash`) for training and for Doc2Vec query inference. This makes the vectors identical in every run (tested in `test_extras.py`). Each model is saved with a `.fingerprint` file (SHA-256 of the corpus and the parameters); if the data or the parameters change, the stale model is retrained automatically.

The Doc2Vec query inference (`infer_query_vector`) runs the same algorithm as gensim's `infer_vector` (a test checks that the two give the same vector) but builds the initial vector with the stable hash itself, so it changes no global state. It does reseed `model.random`, so concurrent web requests (Part 4) should not share one model object between threads at the same time. The fingerprint hashes the documents in training order, because training order changes the model.

Every ranker rejects `top_k` values that are not `None` or a positive integer (e.g. `0`, `-1`, `2.5`), and the command line accepts only positive `--top-k` values.

Tested with Python 3.13.3 and the exact versions in `requirements.txt`.

---

## Tests

There are 49 Part 3 tests, and none of them needs the dataset:

- `test_part_3.py` (17): AND candidates shared by all methods; invalid `top_k` rejected by every ranker; TF-IDF identical to Part 2; BM25 against hand-computed values; BM25 saturation and length normalisation; our score; word2vec averaging and cosine; products without a vector ranked last; agreement measures.
- `test_intent_score.py` (8): attribute parsing, gender detection, query intent, penalties for each kind of contradiction, the product of factors, gender-neutral queries.
- `test_extras.py` (24): re-exporting the ablation judgments keeps every existing label and annotation, exact permutation-test p-values, Holm adjustment (hand values, and the column in the significance table), Doc2Vec inference identical to gensim's with a stable hash (needs gensim), Cohen's kappa and the blind agreement sample, RBO values computed by hand, near-tie counting, unjudged products and condensed lists in the ablation, the significance table, embeddings that do not depend on Python's hash seed (needs gensim), the truncated AP@K convention, rejection of invalid K and mismatched lists, judgment-pool preservation (a judged product that leaves every top 20 keeps its label), label-conflict detection, pool and ranking validation, the candidate-set check, unique product IDs, and model fingerprints.

Passing tests check the code, not the quality of the relevance judgments.

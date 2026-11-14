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
- **Pooling:** for each query, the union of every method's top 20 is judged. Part 2 labels are reused automatically, and existing labels are never overwritten. All methods are evaluated against the same set of judged relevant products, so recall, AP and NDCG are comparable across methods.
- **Judging rules:**
    - A product is relevant if it matches the product type, the audience (if stated) and every stated attribute.
    - When fields conflict, they are trusted in this order: title > structured attributes > description > category/URL.
    - A blend counts as the material ("cotton blend" satisfies "cotton").
- **Metrics:**
    - Computed at K = 5, 10, 15 and 20: P@K, R@K, F1@K, AP@K/MAP@K, NDCG@K and RR@K/MRR@K.
    - Paired randomisation tests against BM25 and TF-IDF (exact over all 2^8 sign assignments).
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
python -m project_progress.part_3.ablation
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
| `results/significance.csv` | paired randomisation tests (K = 10) against BM25 and TF-IDF |
| `results/validation_methods*.csv` | every method on the professor's validation labels |
| `results/method_agreement.csv` | overlap@5/10/20, Jaccard@20, RBO@20 and Kendall's tau per method pair |
| `results/score_spread.csv` | scores at ranks 1/10/20 and near-ties around rank 10 |
| `results/tfidf_vs_bm25_rank_shifts.csv` | products that move most between TF-IDF and BM25, with length and term frequencies |
| `results/ablation.csv`, `results/bm25_grid.csv` | ablations of both original scores; BM25 k1 × b grid |
| `results/score_components.csv` | every component of both original scores for each top-20 product |
| `results/word2vec_neighbours.csv` | nearest neighbours of every query term |
| `figures/*.png` | P@10 by query, MAP@K by K, Kendall's tau heatmap |

The trained models are stored in `models/`, which is git-ignored because they can be rebuilt with the same seed (`run_rankings --retrain`).

---

## Tests

There are 31 Part 3 tests, and none of them needs the dataset:

- `test_part_3.py` (15): AND candidates shared by all methods; TF-IDF identical to Part 2; BM25 against hand-computed values; BM25 saturation and length normalisation; our score; word2vec averaging and cosine; agreement measures.
- `test_intent_score.py` (8): attribute parsing, gender detection, query intent, penalties for each kind of contradiction, the product of factors, gender-neutral queries.
- `test_extras.py` (8): exact permutation-test p-values, RBO values computed by hand, near-tie counting, unjudged products in the ablation, and the significance table.

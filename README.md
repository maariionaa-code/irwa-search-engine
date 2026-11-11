# Information Retrieval and Web Analytics (IRWA) - Final Project

This repository contains our implementation of the **Information Retrieval and Web Analytics (IRWA) Final Project**.

The project develops an information retrieval system over a fashion-product dataset and is implemented incrementally in four parts:

- **Part 1 — Text Processing and Exploratory Data Analysis**
- **Part 2 — Indexing and Evaluation**
- **Part 3 — Ranking and Filtering**
- **Part 4 — RAG, Web Interface and Web Analytics**

The project is implemented in Python. The final stage will integrate the search engine into a Flask-based web application.

---

## Authors

- Mariona Espí U232616
- Ander Barnadas U233729
- Pau Borrell U207992

Repository:

```text
https://github.com/maariionaa-code/irwa-search-engine
```

---

# Project Structure

```text
irwa-search-engine/
├── data/
│   ├── fashion_products_dataset.json
│   └── validation_labels.csv
│
├── myapp/
│   ├── analytics/
│   ├── core/
│   ├── generation/
│   └── search/
│
├── project_progress/
│   ├── part_1/
│   │   ├── part_1.ipynb
│   │   ├── preprocessing.py
│   │   ├── test_preprocessing.py
│   │   ├── README.md
│   │   ├── IRWA-2026-u232616-u233729-u207992-part-1.pdf
│   │   └── figures/
│   │
│   ├── part_2/
│   │   ├── indexing.py
│   │   ├── ranking.py
│   │   ├── evaluation.py
│   │   ├── evaluate_validation.py
│   │   ├── evaluate_validation_ranked.py
│   │   ├── prepare_test_queries.py
│   │   ├── evaluate_test_queries.py
│   │   ├── recalculate_tfidf_scores.py
│   │   ├── test_part_2.py
│   │   ├── test_queries_judgments.csv
│   │   ├── test_queries_metrics.csv
│   │   ├── test_queries_summary.csv
│   │   └── validation_ranked_metrics.csv
│   │
│   └── part_3/
│       ├── corpus.py
│       ├── ranking_methods.py
│       ├── embedded_ranking.py
│       ├── pipeline.py
│       ├── queries.py
│       ├── query_term_stats.py
│       ├── run_rankings.py
│       ├── export_judgment_pool.py
│       ├── evaluate_methods.py
│       ├── compare_rankings.py
│       ├── make_figures.py
│       ├── test_part_3.py
│       ├── judgments_pool.csv
│       ├── README.md
│       ├── results/                  
│       ├── figures/                  
│       └── models/           
│ 
├── static/
├── templates/
├── .gitignore
├── LICENSE
├── requirements.txt
├── web_app.py
└── README.md
```

The `project_progress/` directory contains the implementation and report material corresponding to each incremental project submission.

---

# Project Progress

| Part | Description | Status |
|---|---|---|
| Part 1 | Text Processing and Exploratory Data Analysis | Complete |
| Part 2 | Indexing and Evaluation | Complete |
| Part 3 | Ranking and Filtering | Complete |
| Part 4 | RAG, Web Interface and Web Analytics | Not completed yet |

---

# Data

The project uses the fashion-products dataset provided for the course.

Before running the project, place the following files inside the root `data/` directory:

```text
data/
├── fashion_products_dataset.json
└── validation_labels.csv
```

These dataset files are not included in the repository.

The main dataset contains product information such as:

- product ID
- title
- description
- brand
- category
- sub-category
- product details
- seller
- stock availability
- selling price
- actual price
- discount
- average rating
- product URL

The validation-label file contains relevance judgements that are used in later stages of the project for search-engine evaluation.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/maariionaa-code/irwa-search-engine.git
cd irwa-search-engine
```

---

## 2. Create a virtual environment

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Download required NLTK resources

The preprocessing pipeline uses the NLTK English stop-word corpus.

Run:

```bash
python -m nltk.downloader stopwords
```

---

## 5. Add the data files

Make sure these two files exist:

```text
data/fashion_products_dataset.json
data/validation_labels.csv
```

---

# Part 1 — Text Processing and Exploratory Data Analysis

Part 1 prepares the fashion-product corpus for later retrieval and ranking stages.

The implementation includes:

- dataset inspection
- missing-value analysis
- text normalization
- tokenization
- punctuation removal
- stop-word removal
- stemming
- product metadata preprocessing
- structured `product_details` processing
- numerical field conversion
- construction of searchable document representations
- preprocessing unit tests
- exploratory data analysis
- corpus vocabulary analysis
- category, brand and seller analysis
- stock, price, discount and rating analysis
- generated visualizations

The original product fields are preserved for display purposes while additional processed representations are created for future indexing and ranking.

Detailed Part 1 documentation is available in:

```text
project_progress/part_1/README.md
```

---

# Running Part 1

From the repository root, first run the preprocessing tests:

```bash
python -m pytest project_progress/part_1/test_preprocessing.py -v
```

All tests should pass.

Then launch the Part 1 notebook:

```bash
jupyter lab project_progress/part_1/part_1.ipynb
```

Alternatively, the notebook can be opened directly in VS Code or another Jupyter-compatible environment.

Run all notebook cells from top to bottom.

The generated EDA figures are stored in:

```text
project_progress/part_1/figures/
```

---

# Part 1 Search Representation

The Part 1 preprocessing pipeline preserves the original product information and generates separate processed fields for:

- `title`
- `description`
- `brand`
- `category`
- `sub_category`
- `product_details`
- `seller`

A combined `search_tokens` representation is constructed using:

- title
- description
- brand
- category
- sub-category
- product details

The seller field is retained separately but is not included in the default combined representation.

Numerical and boolean attributes such as:

- selling price
- actual price
- discount
- average rating
- stock availability

are maintained as structured values rather than being indexed as ordinary textual terms.

These decisions provide the basis for the indexing and ranking methods implemented in later parts of the project.

---


# Part 2 — Indexing and Evaluation

Part 2 implements an inverted index, conjunctive retrieval
(AND), TF-IDF ranking and retrieval evaluation over the
fashion-product corpus prepared in Part 1.

## Implementation

The implementation is located in:

`project_progress/part_2/`

The main components are:

- `indexing.py`: loads processed documents, constructs an
  inverted index and implements Boolean AND retrieval.
- `ranking.py`: ranks matching products using TF-IDF
  cosine similarity.
- `evaluation.py`: implements the seven evaluation metrics.
- `evaluate_validation.py`: evaluates the original order
  of the professor's labeled validation documents.
- `evaluate_validation_ranked.py`: evaluates TF-IDF within
  the professor's judged document pool.
- `prepare_test_queries.py`: generates ranked candidates
  for five custom queries.
- `evaluate_test_queries.py`: evaluates the manually
  labeled custom-query results.
- `test_part_2.py`: automated tests for Part 2.

## Retrieval Methodology

Documents and queries use the same preprocessing pipeline
as Part 1, including normalization, tokenization,
stop-word removal and Snowball stemming.


We also added an explicit normalization rule for the
irregular plural "women", which is converted to "woman"
before applying Snowball stemming.

This correction was necessary because Snowball does not
automatically normalize this irregular plural.

By applying the same rule to documents and queries,
both "women" and "woman" can match the same products.

The inverted index maps each term to the product IDs
containing that term.


The default `search_tokens` representation combines the
processed title, description, category, sub-category and
product details.

Brand and seller tokens are preserved separately but
excluded from the default searchable representation.
This avoids introducing irrelevant matches or giving
unnecessary importance to brand and seller names
during TF-IDF ranking.

The original fields remain available for future
field-specific searches and filtering.

Boolean AND retrieval requires every processed query
term to appear in the searchable document representation.

TF-IDF ranking uses:

- Logarithmic term frequency: 1 + ln(tf)
- Smoothed IDF: ln((N + 1) / (df + 1)) + 1
- Cosine similarity between document and query vectors

Results are sorted by decreasing cosine similarity.

## Running Part 2

Run all commands from the repository root.

### Inverted index and AND retrieval

```bash
python -m project_progress.part_2.indexing
```

### TF-IDF ranking

```bash
python -m project_progress.part_2.ranking
```

### Original validation-label evaluation

```bash
python -m project_progress.part_2.evaluate_validation
```

### TF-IDF validation within the labeled pool

```bash
python -m project_progress.part_2.evaluate_validation_ranked
```

### Five custom queries

The evaluated queries are:

- Q1: women cotton kurta
- Q2: men slim fit jeans
- Q3: men running shoes
- Q4: women full sleeve jacket
- Q5: men cotton sports shorts

The file `test_queries_judgments.csv` contains the
top 20 ranked products for each query, with binary
relevance judgments.

To calculate the evaluation metrics:

```bash
python -m project_progress.part_2.evaluate_test_queries
```

This produces:

- `test_queries_metrics.csv`
- `test_queries_summary.csv`

The query candidate exporter refuses to overwrite an
existing judgments file, preventing accidental loss
of manual relevance labels.

## Evaluation Metrics

The implemented metrics are:

- Precision@K
- Recall@K
- F1@K
- Average Precision@K
- Mean Average Precision (MAP)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

Results are evaluated for K = 5, 10 and 20.

Relevance judgments are binary: 1 for relevant
documents and 0 for non-relevant documents.

Recall and related metrics for the custom queries are
relative to the 20 labeled candidates per query.
They should not be interpreted as full-corpus recall.

For the professor's validation queries, TF-IDF is also
evaluated within the available labeled pool. Unjudged
documents are excluded, so these results are not
equivalent to full-corpus top-K evaluation.

## Tests

Run all Part 1 and Part 2 tests:

```bash
python -m pytest project_progress/part_1/test_preprocessing.py project_progress/part_2/test_part_2.py -v
```

The verified test run passed all 33 tests:
26 preprocessing tests and 7 Part 2 tests.

# Part 3 — Ranking and Filtering

Part 3 keeps the conjunctive (AND) retrieval of Part 2 fixed and compares different ways of ordering the same matching products. Every method ranks exactly the same candidates, so differences between methods come only from the scoring function.

## Ranking methods
- TF-IDF + cosine (ranking_methods.rank_tfidf): Same formulation as Part 2 - (verified by a test).
- BM25 (ranking_methods.rank_bm25):Saturated term frequency (k1=1.2) and length normalization relative to the average (b = 0.75).
- Our score (ranking_methods.rank_our_score): 
- word2vec + cosine (embedded_ranking.rank_word2vec): Average of word vectors trained on our corpus, cosine with the query
- Doc2Vec + cosine (embedded_ranking.rank_doc2vec): Additional experiment, learned document vectors (PV-DBOW)

## Implementation
The implementation is located in project_progress/part3/:
- corpus.py: loads the dataset with Part1 preprocessing and builds the shared index statistics (term frequencies, document frequencies, document lengths)
- ranking_methods.py: TF-IDF, BM25 and our own score.
- embedded_ranking.py: trains word2vec and Doc2Vec (fixed seed, single worker for reproducibility) and ranks with them.
- pipeline.py: Search Engine class exposing every method, plus a command-line search.
- queries.py: the five Part 2 queries (Q1-Q5) and three gender neutral queries (Q6-Q8).
- query_term_stats.py: doucment frequency and IDF of every query term.
- run_rankings.py: top-20 list per query and method.
- export_judgment_pool.py: builds the judgment pool (union of all methods top 20), reusing Part 2 labels and never overwriting existing ones.
- evaluate_methods.py: all evaluation metrics per method, reusing the Part 2 metric functions.
- compare_renking.py: agreement between methods (overlap@10, Jaccard@20, Kendalls tau) and the products whose rank changes most between TF-IDF and BM25.
- make_figures.py: figures for the report
- test_part_3.py: automated tests for Part 3, including hand-computed BM25 values.

Detailed Part 3 documentation is available in project_progress/part_3/README.md

## Running Part 3
Run all commands from the repository root, in this order:
python -m project_progress.part_3.query_term_stats      # 1. query term statistics
python -m project_progress.part_3.run_rankings          # 2. top-20 lists (trains the embeddings on first run)
python -m project_progress.part_3.export_judgment_pool  # 3. judgment pool to label
python -m project_progress.part_3.evaluate_methods      # 4. evaluation metrics per method
python -m project_progress.part_3.compare_rankings      # 5. agreement between methods
python -m project_progress.part_3.make_figures          # 6. figures

To search with any query and method:
python -m project_progress.part_3.pipeline "men slim fit jeans" --method our_score --top-k 10

Available methods: tfidf, bm25, our_score, word2vec, doc2vec.
The trained models are saved in project_progress/pert3/models/, which is not committed because the models can be reproduced with the same seed. Use python -m project_progress.part_3.run_rankings --retrain to rebuild them.
---

# Part 4 — Web Application and RAG

The final stage of the project will integrate the search engine into a Flask-based application.

Part 4 will include:

- search-engine integration
- user interface
- Retrieval-Augmented Generation (RAG)
- interaction with an LLM
- web analytics
- query and user-interaction tracking

The final web application will use the existing:

```text
web_app.py
myapp/
templates/
static/
```

structure.

Detailed execution instructions for the final application will be added once Part 4 is completed.

---

# Testing

Part 1 preprocessing tests can be executed from the repository root with:

```bash
python -m pytest project_progress/part_1/test_preprocessing.py -v
```

Additional tests may be added as later parts of the project are implemented.

---

# Reproducibility

To reproduce the current Part 1 results:

```bash
git clone https://github.com/maariionaa-code/irwa-search-engine.git
cd irwa-search-engine

python -m venv .venv
```

Activate the environment.

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Then run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

Add:

```text
data/fashion_products_dataset.json
data/validation_labels.csv
```

Then:

```bash
python -m pytest project_progress/part_1/test_preprocessing.py -v
jupyter lab project_progress/part_1/part_1.ipynb
```

Run the notebook from beginning to end to reproduce the preprocessing results, statistics and EDA figures.

---

# AI Usage

Generative AI tools were used during the development process for tasks such as:

- reviewing preprocessing decisions
- assisting with code generation
- suggesting test cases
- reviewing project structure
- improving documentation clarity

All suggestions incorporated into the project were reviewed and understood by the group. The group remains responsible for the final code, methodology, analysis and conclusions.
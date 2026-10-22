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
│   └── part_3/
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
| Part 2 | Indexing and Evaluation | Not completed yet |
| Part 3 | Ranking and Filtering | Not completed yet |
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

# Parts 2 and 3

Parts 2 and 3 will extend the preprocessing work with information-retrieval functionality including:

- inverted indexes
- conjunctive retrieval
- TF-IDF
- ranking
- retrieval evaluation
- BM25
- alternative ranking strategies
- filtering
- semantic representations

Their implementations will be stored under:

```text
project_progress/part_2/
project_progress/part_3/
```

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
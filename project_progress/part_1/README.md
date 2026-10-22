# IRWA Project: Part 1  
## Text Processing and Exploratory Data Analysis

This folder contains the implementation of **Part 1** of the IRWA final project.

The objective of this part is to prepare the fashion-product dataset for future information retrieval tasks and perform exploratory data analysis to better understand the corpus.

---

## Authors

- Mariona Espí U232616
- Ander Barnadas U233729
- Pau Borrell U207992

GitHub repository: (https://github.com/maariionaa-code/irwa-search-engine)

---

## Folder Structure

```text
project_progress/part_1/
├── part_1.ipynb
├── preprocessing.py
├── test_preprocessing.py
├── README.md
├── IRWA-2026-u232616-u233729-u207992-part-1.pdf
└── figures/
```

### `part_1.ipynb`

Main notebook containing:

- dataset loading and validation
- initial dataset inspection
- missing-value analysis
- text preprocessing demonstrations
- preprocessing of the complete corpus
- metadata and structured-field processing
- numerical-field processing
- exploratory data analysis
- plots and summary statistics used in the report

### `preprocessing.py`

Contains the reusable preprocessing functions used throughout Part 1.

### `test_preprocessing.py`

Contains unit tests for the preprocessing functions.

### `figures/`

Contains the plots generated during the exploratory data analysis.

### `IRWA-2026-u232616-u233729-u207992-part-1.pdf`

Final report for Part 1 containing the main decisions, methodology, results and conclusions.

---

# Data

The project uses the provided fashion-products dataset:

```text
data/fashion_products_dataset.json
```

The validation labels are stored in:

```text
data/validation_labels.csv
```

These files are not committed to Git and must be placed inside the root `data/` folder before executing the notebook.

The dataset contains **28,080 products**.

---

# Text Preprocessing

The following preprocessing pipeline is applied to textual product information:

1. Missing-value handling
2. HTML entity decoding
3. Unicode normalization
4. Lowercasing
5. Possessive normalization
6. Splitting hyphenated and compound terms
7. Tokenization
8. Punctuation removal
9. English stop-word removal
10. Stemming using the NLTK Snowball stemmer

For example:

```text
Women's Full-Sleeve Cotton Sweatshirts
```

is transformed approximately into:

```text
women, full, sleev, cotton, sweatshirt
```

The same preprocessing function should later be applied to user queries so that queries and documents use the same representation.

---

# Searchable Fields

The original product fields are preserved so that they can later be displayed in search results.

Processed representations are created separately for:

- `title`
- `description`
- `brand`
- `category`
- `sub_category`
- `product_details`
- `seller`

For example:

```text
title
title_tokens
```

and:

```text
description
description_tokens
```

are both retained.

This prevents the preprocessing stage from destroying the original human-readable information.

---

# Metadata Processing Decision

We decided to **retain the processed fields separately while also constructing a combined searchable representation**.

The combined `search_tokens` field contains tokens from:

- title
- description
- brand
- category
- sub-category
- product details

This approach was selected because it provides two advantages:

1. A combined representation makes document-level retrieval straightforward.
2. Keeping the fields separately preserves information about where each term appeared, which may later allow field-specific weighting during ranking.

For example, a term appearing in the `brand` field may provide different evidence from the same term appearing only in the description.

Token duplicates are intentionally preserved because term frequency will be required by later ranking methods such as TF-IDF and BM25.

---

# Product Details

The `product_details` field contains useful structured information such as:

- fabric
- colour
- pattern
- fit
- closure
- pockets
- style information

The original structure is preserved, while an additional text representation is created for retrieval.

For example:

```python
[
    {"Fabric": "Cotton Blend"},
    {"Color": "Blue"},
    {"Pattern": "Solid"}
]
```

is converted into text similar to:

```text
Fabric Cotton Blend Color Blue Pattern Solid
```

and subsequently processed using the same text-preprocessing pipeline.

---

# Seller Field

The seller field is processed and preserved separately.

However, seller tokens are **not included in the default combined `search_tokens` representation**.

This decision was made because seller identity generally describes the merchant rather than the product itself.

The field remains available for future seller-specific searches, filtering or ranking experiments.

---

# Numerical and Boolean Fields

The following fields are not indexed as textual terms:

- `selling_price`
- `actual_price`
- `discount`
- `average_rating`
- `out_of_stock`

Instead, they are maintained as structured numerical or boolean information.

Examples:

```text
"2,999"   → 2999.0
"69% off" → 69.0
"3.9"     → 3.9
```

The `out_of_stock` field remains boolean.

These fields may later be useful for:

- filtering
- sorting
- custom ranking functions
- stock-aware search
- price-aware search

Indexing them as ordinary words would lose their numerical meaning.

---

# Exploratory Data Analysis

The notebook includes exploratory analysis of the corpus.

The analysis includes:

- dataset size and missing values
- title-length distribution
- description-length distribution
- searchable-document length distribution
- approximate sentence-length analysis
- processed vocabulary size
- most frequent search terms
- word cloud
- category distribution
- sub-category distribution
- brand distribution
- seller distribution
- stock availability
- selling-price distribution
- discount distribution
- rating distribution
- highly rated products
- most expensive products
- products with the highest discounts
- relationship between selling price and discount

Generated plots are stored in:

```text
project_progress/part_1/figures/
```

---

# Main EDA Findings

The corpus contains **28,080 products** and approximately **20,677 unique processed terms**.

The average processed title contains approximately **6.27 tokens**, while the average description contains approximately **18.84 tokens**.

The most frequent terms include:

- fabric
- neck
- sleev
- shirt
- fit

These terms mainly represent **Clothing characteristics or parts**.

The dataset contains approximately:

- **325 unique brands**
- **535 unique sellers**
- **26,436 in-stock products**
- **1,644 out-of-stock products**

The most represented categories/subcategories are:

- Clothing and Accessories
- Topwear
- Footwear

The category distribution is **strongly imbalanced**.

The median selling price is approximately **545**, the median discount is approximately **53%**, and the median average rating is approximately **3.63**.

---

# Requirements

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The NLTK English stop-word corpus is also required:

```bash
python -m nltk.downloader stopwords
```

---

# Quick Start

From the repository root, the complete sequence to reproduce Part 1 is:

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# macOS/Linux:
# source .venv/bin/activate

# 3. Install project dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Download the NLTK stop-word corpus
python -m nltk.downloader stopwords

# 5. Make sure the required data files are inside:
# data/fashion_products_dataset.json
# data/validation_labels.csv

# 6. Run the Part 1 tests
python -m pytest project_progress/part_1/test_preprocessing.py -v

# 7. Launch Jupyter
jupyter lab project_progress/part_1/part_1.ipynb
```

Once the notebook opens, restart the kernel if necessary and run all cells from top to bottom.

The generated EDA figures will be saved automatically in:

```text
project_progress/part_1/figures/
```

---

# Running Part 1

From the repository root, open:

```text
project_progress/part_1/part_1.ipynb
```

and execute all cells from top to bottom.

The notebook expects the dataset to be available at:

```text
data/fashion_products_dataset.json
```

and the validation labels at:

```text
data/validation_labels.csv
```

Running the notebook generates the EDA figures inside:

```text
project_progress/part_1/figures/
```

---

# Running the Tests

From the repository root run:

```bash
python -m pytest project_progress/part_1/test_preprocessing.py -v
```

Expected result:

```text
24 passed
```

All tests should pass before reproducing or submitting the results.

---

# Part 1 Conclusions

Part 1 produces a clean and reusable representation of the fashion-product corpus for the later information-retrieval stages.

The main design choices are:

- preserve the original product information
- preprocess textual information into separate token fields
- construct an additional combined searchable representation
- include useful structured product details in retrieval
- retain seller information separately
- preserve term frequencies
- maintain prices, discounts, ratings and stock information as structured values rather than textual tokens

These decisions provide a basis for the inverted index, TF-IDF retrieval and evaluation implemented in the following parts of the project.

---

# AI Usage

Generative AI tools were used for **reviewing preprocessing decisions, assisting with code generation, suggesting test cases and checking documentation clarity**.

All generated suggestions were reviewed and understood before being incorporated into the project. The group remains responsible for the final implementation, analysis and conclusions.
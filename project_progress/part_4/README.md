# Part 4 · Retrieval-augmented generation

Run the commands from the repository root with the project's Python environment and the course catalogue at `data/fashion_products_dataset.json`.

## Run the web app

The app reads `GROQ_API_KEY` from the environment or the repository's ignored `.env` file. Set your own Groq key. The default model is `openai/gpt-oss-20b` through Groq. You can set `GROQ_MODEL` to an alternative Groq model that supports JSON object mode. Results and product details remain available if the key is missing or the API fails.

```text
python web_app.py
```

Open `http://127.0.0.1:8088/` and run a product search.

## How the recommendation works

1. **Catalogue evidence.** The model receives product descriptions and structured fields such as fabric, price, rating, and availability. The starter prompt only receives product IDs and titles.
2. **Candidate checks and abstention.** The summary excludes out-of-stock items and compares explicit audience signals in the title, catalogue metadata, and retailer URL. If the search specifies an audience, the available signals must support it. An empty candidate list skips the API call. This conservative heuristic can remove valid items when the catalogue has missing fields or misleading URLs.
3. **Validated selection.** The model selects a product ID and up to three supporting field names. The app verifies the selection and displays the exact catalogue values. This limits unsupported claims about prices and attributes. It does not guarantee that the selected product is the best semantic match or that the catalogue itself is accurate.

These checks apply to the AI summary. The results list preserves the original BM25 order.

## Reproduce the comparison

```text
python -m pytest project_progress/part_4/test_rag.py -q
python -m project_progress.part_4.compare_rag --inspect --output data/rag-context.json
python -m project_progress.part_4.compare_rag
```

The inspect command records which real candidates reach the improved generator without using a key or making API calls. The live command writes `data/rag-comparison.json` with the shared BM25 results, the eligible input for the improved prompt, and both outputs. Both variants use the same Groq model and generation limits. The comparison preserves the starter's prompt and title-only input while replacing its retired default model for both variants.

Review each case for product suitability, whether the cited evidence supports the query, and whether the system abstains when it should. The starter's free-text answer requires human review. An unavailable or invalid response is an integration outcome and should be recorded separately from recommendation quality. Save the live results before writing the report.

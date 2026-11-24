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

## Web analytics

Start the app and use the **Analytics** link. The dashboard starts empty. Submit a product search, open a result, and return to the same results with the browser Back button or the product page's **Back to results** link. Submit an unlikely query to exercise the no-result metric. Reload the dashboard to see the recorded activity. Returning to the same results reuses the stored ranking and summary without requesting Groq again.

### Recorded data

| Record | Grain and fields |
| --- | --- |
| Visit | One browser visit, with anonymous browser ID, last activity, browser, OS, device and optional location. There is a new visit after 30 minutes of inactivity. The browser ID is stored in a signed, 30-day cookie. |
| HTTP request | One GET to home, search or product detail, including visit, UTC time, method, endpoint, HTTP status and elapsed server time. |
| Search | One submitted query, including visit, mission, UTC time, ordered processed terms, ranked product IDs and the displayed summary. |
| Product click | One attributed detail-page open, including visit, search ID, product ID, 1-based rank, source and dwell seconds if measured. Opening a product directly still counts as a view in Quick Stats. |
| Search mission | Similar queries from the same browser. We match the last query in a mission from the past seven days by exact normalized text or at least two shared processed terms with Jaccard similarity of at least 0.5. This is a heuristic, not a labelled measure of user intent. |

The dashboard filters by a search's UTC submission time for search metrics and by request time for HTTP metrics. Clicks belong to the selected search cohort even if the shopper clicked later. Click-through is the share of searches with at least one attributed detail click. No-result rate is the share with zero hits. The dwell median uses completed returns within 30 minutes; a missing return has no dwell value. The rank chart counts detail opens, including reloads, and excludes retailer links. A multi-visit mission has search events in at least two visits within the selected period.

Events and search snapshots live in one Flask process and reset when it restarts. Run the local single-worker app when demonstrating analytics. Device, browser and operating-system labels are inferred from the user agent. The optional setting `IRWA_GEOIP_CSV` accepts a local CSV with `network,country,city` columns and IP networks in CIDR notation. It uses the connection address to look up country and city without saving the address or making an external request. With the setting absent, no location chart is shown.

### Test analytics

```text
python -m pytest project_progress/part_4 -q
```

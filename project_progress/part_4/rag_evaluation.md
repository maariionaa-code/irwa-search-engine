# Live RAG comparison

The [frozen live responses](rag-comparison.json) cover the 12 queries in the [evaluation protocol](rag_evaluation_protocol.md). Both variants use the same initial BM25 top 20 and `openai/gpt-oss-20b` on Groq because the starter's original default was retired for the free tier. The original text of the starter prompt is preserved. One baseline request failed at the transport layer, was retried on its own, and is recorded in the snapshot metadata.

## Per-query judgements

Each triple is **suitable product / correct evidence / correct abstention**. Y means yes and N means no. Correct abstention refers to deciding whether to recommend any item. A response can correctly decide to recommend something yet pick an unsuitable item. An abstention scores N for product and evidence because it supplied neither.

| Query | BM25 / eligible | Starter PID and BM25 rank | Starter | Improved PID and BM25 rank | Improved |
| --- | ---: | --- | :---: | --- | :---: |
| women cotton kurta | 20 / 0 | `KTAFZ8JECS48J3AG` (1) | N / N / N | Abstain | N / N / Y |
| men running shoes | 20 / 14 | `SHOESS9ZEZ6TEQUJ` (2) | Y / N / Y | `SHOFTZFPNCU4VWHK` (17) | Y / Y / Y |
| men slim fit jeans | 20 / 20 | `JEAFDWG9K7KPKHS8` (8) | Y / Y / Y | `JEAEHGRJSGGYEYYX` (1) | Y / Y / Y |
| zzzxxyynoresults | 0 / 0 | Abstain | N / N / Y | Abstain | N / N / Y |
| women full sleeve jacket | 20 / 0 | `JCKFMV5WTASYK4Y2` (1) | N / N / N | Abstain | N / N / Y |
| men cotton sports shorts | 20 / 19 | Abstain | N / N / N | `SRTEQ4873ZRGBPYJ` (1) | Y / Y / Y |
| black track pants | 20 / 14 | `TKPFGGWJMDQFKZ5Y` (1) | N / N / Y | `TKPFZMQ8YMWNGXEU` (2) | Y / Y / Y |
| hooded sweatshirt | 20 / 10 | Abstain | N / N / N | `SWSFXZWGFKHNG88S` (1) | Y / Y / Y |
| denim jacket | 20 / 7 | `JCKFURYWGYKVKFH3` (3) | N / Y / Y | `JCKFWC5YZKFWXGEB` (13) | Y / Y / Y |
| women full sleeve sweatshirt cotton | 20 / 0 | Abstain | N / N / Y | Abstain | N / N / Y |
| men slim jeans blue | 20 / 13 | `JEAFSKYHRVZSABPR` (2) | Y / Y / Y | `JEAFSKYHRVZSABPR` (2) | Y / Y / Y |
| men running shoes under 1000 | 0 / 0 | Abstain | N / N / Y | Abstain | N / N / Y |
| **Yes totals** | | | **3 / 3 / 8** | | **7 / 7 / 12** |

## Decisions that affect the scores

- **Women’s garments.** Each of the three pools contains titles or audience fields that disagree with the retailer path, plus some unavailable items. We treated unresolved audience contradictions as insufficient for a defensible recommendation. The rule is conservative and can reject usable products if the retailer path is wrong.
- **Shoes.** Both selections are in-stock men’s running shoes. The starter infers performance and comfort for a product with an empty description. The improved output displays the registered brand, selling price and discount. Those facts support a shopping comparison, while the title establishes the running use and audience.
- **Sports shorts.** The displayed fabric is Cotton Blend, which we counted as meeting a request for cotton without a purity requirement. This does not establish that the product is pure cotton.
- **Track pants and denim jacket.** The starter’s main picks are out of stock. The track-pants title also conflicts with its retailer slug. The denim jacket’s title and price support the item description even though it cannot be recommended as available. The improved picks are in stock and match the product requests.
- **Hooded sweatshirt.** The source field says `Hooded: Yes`. The visible summary cites the category, pattern and stock, with the title establishing that it is a sweatshirt. This passes the pre-set evidence rule, which requires one supported query-relevant claim. The displayed facts still omit the decisive hood attribute.
- **Price-constrained query.** The shared lexical pool was empty. Both variants were right to abstain on that input. The outcome does not establish that the catalogue contains no in-stock running shoes below 1000.

These are catalogue-based judgements on one run, not an estimate of general accuracy. See the protocol for definitions and the raw JSON for complete responses. [Figure 4 data](figure4_counts.json) is checked against the frozen response file by the offline test suite.

## Reproduce

From the repository root, set up the catalogue and Groq key as described in the [Part 4 guide](README.md), then run:

```bash
python -m project_progress.part_4.compare_rag --output data/rag-comparison-recheck.json
```

The default query set matches the archived evaluation. Use `--inspect` to check the retrieved and eligible lists without an API key. Model outputs can differ when rerun.

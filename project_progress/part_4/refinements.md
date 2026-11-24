# Checks after the frozen RAG run

These checks use the course catalogue and no provider calls. The [archived live run](rag-comparison.json), its judgements and Figure 4 retain their original values. The command below reproduces the [new check output](refinement_checks.json).

```bash
python -m project_progress.part_4.inspect_refinements
```

## Price restrictions and the web ranking

`men running shoes under 1000` had zero hits in the frozen run. The revised engine searches for `men running shoes`, retains results whose recorded selling price is strictly below 1000 and only then takes the top 20. The check returned **20** products with prices from **249 to 990**. Strict RAG eligibility retains **12**. A price filter applies one constraint. It does not resolve the unrelated products that still pass lexical matching.

The existing top-five comparison for `men running shoes` is reproduced against the current catalogue:

| Rank | BM25 | Our Score |
| ---: | --- | --- |
| 1 | Cricket Shoes For Women | Running Shoes For Men (Multicolor) |
| 2 | Running Shoes For Men (Multicolor) | Running Shoes For Men (Multicolor) |
| 3 | Running Shoes For Men (Multicolor) | SPRINT Running Shoes For Men (Black, Grey) |
| 4 | Running Shoes For Women (White) | SPRINT Running Shoes For Men (Blue, Grey) |
| 5 | SPRINT Running Shoes For Men (Black, Grey) | RS-7200 Running Shoes For Men (Navy) |

Our Score improves this example. Part 3’s five inherited queries gave mean precision@10 of 0.960 for BM25 and 1.000 for Our Score on the judged pool. The web demo keeps BM25 as the lexical reference so relevance order remains comparable with the baseline. Our Score blends additional catalogue signals and can trade text relevance for business utility. The five-query result does not establish general superiority or comparable latency.

## Isolating the retailer-URL rule

The two policies inspect the same ranked pool. Strict is the default. The experimental override requires agreement between Title and Ideal For and flags a disagreement with the URL. The table uses retained relevant products divided by relevant products already present in the retrieved list. Coverage is conditional on existing Part 3 judgements.

| Query | Strict → experimental eligibility | Strict → experimental relevant coverage | Strict exclusion causes |
| --- | ---: | ---: | --- |
| women cotton kurta | 0 → 12 | 0/20 → 12/20 | 20 URL conflicts |
| women full sleeve jacket | 0 → 13 | 0/20 → 13/20 | 13 URL conflicts, 7 out of stock |
| women full sleeve sweatshirt cotton | 0 → 0 | No Part 3 labels for the exact query | 12 URL conflicts, 3 out of stock, 5 audience mismatches |
| denim jacket | 7 → 13 | 7/20 → 13/20 | 6 URL conflicts, 7 out of stock |

Every retained item in the three labelled rows was judged relevant in Part 3. Those labels were for retrieval relevance, so they cannot establish the true audience behind a stale or contradictory URL. The warning is visible if the experimental mode recommends an affected item. This check isolates candidate selection only. It uses no new model answers.

## Evidence and analytics

The evidence check uses a controlled model reply to isolate presentation. If the reply chooses a sweatshirt and cites its category, an explicit hooded query puts the catalogue’s verified Hooded field first when available. A response with an invalid product ID or evidence field is still rejected.

Retailer links resolve through a local route, which counts a separate outbound event for a product owned by the displayed search before redirecting to its catalogue URL. It does not change detail-open counts, CTR or dwell. Dwell is time until return and can include idle time. For a public deployment over HTTPS, enable the optional analytics password and secure cookies described in the [guide](README.md).

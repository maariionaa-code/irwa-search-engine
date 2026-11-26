# Part 4 live follow-up

On 26 November 2026 we reran the twelve fixed queries on the revised Part 4 app at commit `b6ba999`. Both variants received the same initial top 20 from Our Score, with the same price rules, model (`openai/gpt-oss-20b` on Groq), temperature and completion limit. The starter received product IDs and titles. The improved system filtered candidates, supplied catalogue evidence and validated the reply before display. The earlier BM25 comparison remains a separate run.

## Per-query results

Each triple gives suitability, evidence and correct recommendation or abstention, in that order. For this table, contradictory audience evidence counts against a match when the query specifies an audience. For a gender-neutral query we judge the requested attributes directly. Cotton blends count for a request for cotton unless the query asks for pure cotton. We assess only the main recommendation. A speculative factual claim about price or features does not earn evidence credit, even when another claim in the explanation is supported.

| Query | Hits / eligible | Starter product (rank) | Starter | Improved product (rank) | Improved |
| --- | ---: | --- | :---: | --- | :---: |
| women cotton kurta | 20 / 0 | `KTAFZFX7HTMGQCG6` (4) | N / N / N | Abstain | N / N / Y |
| men running shoes | 20 / 19 | `SHOFN3EJVGC9VXYT` (5) | Y / N / Y | `SHOESS9ZEZ6TEQUJ` (1) | Y / Y / Y |
| men slim fit jeans | 20 / 20 | `JEAFDWG9K7KPKHS8` (1) | Y / N / Y | `JEAEVJGUSXRNSHRY` (11) | Y / Y / Y |
| zzzxxyynoresults | 0 / 0 | Abstain | N / N / Y | Abstain | N / N / Y |
| women full sleeve jacket | 20 / 0 | `JCKFUUSN3YZPEJXP` (1) | N / N / N | Abstain | N / N / Y |
| men cotton sports shorts | 20 / 20 | Abstain | N / N / N | `SRTFDT7FAGUTMSVY` (3) | Y / Y / Y |
| black track pants | 20 / 10 | `TKPFYTQEUHRYQRZY` (1) | Y / N / Y | `TKPFZMQ8YMWNGXEU` (2) | Y / Y / Y |
| hooded sweatshirt | 20 / 11 | Abstain | N / N / N | `SWSFY5ZHEJ2HYWDG` (3) | Y / Y / Y |
| denim jacket | 20 / 8 | `JCKFKKFZ7QB2KUQV` (1) | Y / N / Y | `JCKFKKFZ7QB2KUQV` (1) | Y / Y / Y |
| women full sleeve sweatshirt cotton | 20 / 0 | Abstain | N / N / Y | Abstain | N / N / Y |
| men slim jeans blue | 20 / 20 | `JEAFESND4QWQUBZD` (1) | Y / Y / Y | `JEAFESND4QWQUBZD` (1) | Y / Y / Y |
| men running shoes under 1000 | 20 / 15 | `SHOFN3EJVGC9VXYT` (3) | Y / N / Y | `SHOEKNRJ34UCMAPA` (1) | Y / Y / Y |
| **Yes totals** | | | **6 / 1 / 8** | | **8 / 8 / 12** |

The evidence judgments follow the saved replies and source records. The starter added unsupported comfort or construction claims for the kurta and jacket, cushioning and mesh claims for the running shoe, a comparative price claim for the slim-fit jeans, and unverified details about the colour placement and flexibility of the black-and-blue track pants. The track pants still qualify for the gender-neutral request because the source confirms black as one of their colours. Both variants found the same available denim jacket, but the starter called the fit relaxed and the catalogue records it as slim. The blue-jeans reply correctly identified a single pair.

The improved system recovered two cases where the starter abstained. The shorts have Cotton Blend fabric and a Sports Shorts style. The sweatshirt has `Hooded: Yes`, which now appears in the visible summary. For the price query both picks satisfy the threshold, at 990 and 490 in catalogue units. The starter guessed at a typical dollar price range while the improved reply displayed the recorded selling price.

## Sensitivity to audience judgments

In the kurta and jacket cases, both source titles and explicit `Ideal For` fields say Women, while retailer paths contain Men. The strict system abstained on both. If explicit title and catalogue agreement overrides the retailer path, both starter selections become suitable. Under that judgment the starter and improved system tie at **8/12** for suitability and **10/12** for the recommendation decision. Evidence remains **1/12** and **8/12**. The sweatshirt case stays unresolved because its records lack corroborating `Ideal For` fields. Offline inspection of the same retrieved pools made 17 kurtas and 20 jackets eligible under the override, compared with zero under strict mode. Those are candidate counts. The live run used the strict policy throughout.

All twelve starter calls completed. The improved system made eight calls and validated all eight replies. It skipped four calls with empty eligible pools, including the zero-hit query. The [live archive](rag-comparison-current-live.json) includes the ordered retrieval pools, exclusions, raw replies and validated summaries. These are judgments on a small fixed query set, and the authors should review them before incorporating the results into the PDF.

To repeat the live comparison from the repository root with a configured Groq key:

```bash
python -m project_progress.part_4.compare_rag --ranker our_score --pause 18 --output data/rag-comparison-recheck.json
```

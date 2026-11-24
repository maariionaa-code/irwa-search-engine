# Live RAG comparison

One live run used the course fashion catalogue and `openai/gpt-oss-20b` on Groq for both variants. The starter received the IDs and titles of up to 20 BM25 hits. The improved variant started with the same hits, filtered its recommendation candidates, and provided structured catalogue evidence. Its displayed facts were checked against the chosen product record.

| Query | Retrieved / eligible | Starter output | Improved output |
| --- | ---: | --- | --- |
| `women cotton kurta` | 20 / 0 | Recommended beige kurta `KTAFZ8JECS48J3AG` and suggested a yellow variant. | Abstained before any API call. |
| `men running shoes` | 20 / 14 | Recommended `SHOESS9ZEZ6TEQUJ`, a men's running shoe, and inferred performance and comfort from its title. | Recommended the same product and displayed its catalogue category, audience and stock status. |
| `men slim fit jeans` | 20 / 20 | Recommended `JEAFDWG9K7KPKHS8`, a slim men's jean, and suggested a multipack. | Recommended the same product and displayed its catalogue category, audience and washed style. |
| `zzzxxyynoresults` | 0 / 0 | Called the model, which abstained. | Abstained before any API call. |

## Observations

- The beige kurta's title and audience field say **Women**, but its description refers to “him” and its retailer URL contains `men-solid-straight-kurta`. Abstaining avoided recommending a product with conflicting evidence. An erroneous or stale retailer URL can also exclude a suitable item, and the eligibility result covers only the retrieved 20 products.
- The first BM25 hit for the running-shoe query was a women's cricket shoe. It was excluded from the improved summary's candidate set. Both variants selected the same suitable men's shoe in this run, so the observed gains were in the available evidence and the claims shown to the user.
- The two positive queries had the same main product selection in both variants. The improved version cited structured fields that were absent from the title-only prompt. Some of the fields it chose were general, such as category, and gave limited additional support for the specific fit or sport.
- Both variants abstained for the query with zero hits. The improved version made no API call in that case or when all retrieved products were ineligible.

This four-query comparison illustrates behavior and trade-offs. It cannot estimate an overall RAG accuracy gain or isolate the effect of each change. The catalogue can contain inaccurate information and model replies can vary on later runs.

## Reproduce

Run from the repository root with the course catalogue and a Groq key configured as described in the [Part 4 guide](README.md):

```bash
python -m project_progress.part_4.compare_rag
```

Review `data/rag-comparison.json` for the BM25 hits, eligible context and separate baseline and improved responses. Use the `--inspect` flag to reproduce the inputs without calling Groq.

# Part 4 RAG evaluation protocol

Fix this protocol before obtaining the new live responses. Both variants start from the same ordered BM25 top 20 over the course catalogue and use the same Groq model and decoding limits. Evaluate the main recommendation only. Record separate failure statuses for provider errors and invalid responses.

## Decision rules

| Criterion | Yes | No |
| --- | --- | --- |
| Suitable product | The main recommended product belongs to the shared retrieval pool, is available, and the catalogue supports all explicit constraints in the query, including product type, audience, colour, fabric, fit or price where specified. | It violates a constraint, has insufficient evidence for an explicit requirement, is out of stock, or the response abstains. |
| Correct evidence | The response makes at least one query-relevant factual claim about the main product and all factual claims about that product are supported by its catalogue record. The title itself is evidence for attributes it explicitly states. | It makes a factual claim absent from the record, supplies no relevant supporting claim, or abstains. |
| Correct abstention | The response abstains when no defensible match exists in the shared BM25 pool, or issues a recommendation when at least one defensible match exists. | It makes a recommendation when all candidates are unsuitable, abstains when a defensible match exists, or the generation failed. |

Abstention gets No for suitability and evidence because no product or supporting claim was produced. Use the decision column to score correct abstentions. Availability and audience conflicts are adjudicated using the source record, not a variant’s internal eligibility flag. Where evidence is contradictory or a key field is absent, explicitly note the uncertainty and document the adjudication. Do not interpret the cases as a random sample of traffic.

## Fixed queries

1. women cotton kurta
2. men running shoes
3. men slim fit jeans
4. zzzxxyynoresults
5. women full sleeve jacket
6. men cotton sports shorts
7. black track pants
8. hooded sweatshirt
9. denim jacket
10. women full sleeve sweatshirt cotton
11. men slim jeans blue
12. men running shoes under 1000

"""Queries evaluated in Part 3.

Q1-Q5 are the five test queries defined in Part 2 (required by the
assignment). Following the Part 2 feedback ("all five contain 'women' or
'men'"), we add three supplementary queries that do NOT state a gender, so
that we can also analyse how the rankers behave when the audience is open.
The supplementary queries were chosen from frequent product types; their
document frequencies are printed by ``query_term_stats.py``.
"""

PART2_QUERIES = {
    "Q1": "women cotton kurta",
    "Q2": "men slim fit jeans",
    "Q3": "men running shoes",
    "Q4": "women full sleeve jacket",
    "Q5": "men cotton sports shorts",
}

GENDER_NEUTRAL_QUERIES = {
    "Q6": "black track pants",
    "Q7": "hooded sweatshirt",
    "Q8": "denim jacket",
}

ALL_QUERIES = {**PART2_QUERIES, **GENDER_NEUTRAL_QUERIES}

TOP_K = 20
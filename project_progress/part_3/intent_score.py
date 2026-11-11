"""Part 3: our second original score, the INTENT-AWARE score.

Idea: first understand what the user wants from the query, then reward
products that agree with that intent and penalise those that contradict it.
Three signals are inferred, each one motivated by an error we observed:

1. Category coherence (pseudo-relevance feedback). The most common
   sub-category among the top-20 BM25 results is taken as the category the
   user wants (e.g. "Bottomwear" for "slim fit jeans"). Products from other
   sub-categories are demoted. Fixes shirts ranked first for a jeans query.
2. Gender consistency. If the query names a gender ("men", "women"), a
   product whose "Ideal For" field (or title) states another audience is
   demoted. Fixes "Cricket Shoes For Women" ranked first for
   "men running shoes".
3. Attribute agreement. If a query word is a possible value of a structured
   attribute (e.g. "slim" is a value of Fit, "cotton" of Fabric, "jean" of
   Generic Name), we check the product's attribute: agreement keeps the
   score, contradiction (Fit: Jogger for "slim") lowers it. Fixes jogger and
   tapered jeans matching "slim" through their descriptions.

The final score is a PRODUCT of evidence factors, each in (0, 1]:

    intent = T * K * G * A * S

T = BM25 / best BM25 for the query; K, G, A as above; S = stock factor.
A product that agrees with every inferred intent keeps its text score T;
each contradiction multiplies it by a penalty. The product form means a
strong text match cannot compensate for, e.g., the wrong gender: every
condition must hold, like the conjunctive (AND) logic of the retrieval.
"""

from collections import Counter, defaultdict
from functools import lru_cache

import pandas as pd

from project_progress.part_1.preprocessing import preprocess_text
from project_progress.part_3.corpus import SearchIndex
from project_progress.part_3.ranking_methods import (
    OUT_OF_STOCK_FACTOR,
    bm25_scores,
)


# Categorical attributes whose values describe the product.
ATTRIBUTE_KEYS = {
    "fabric", "top fabric", "bottom fabric", "outer material",
    "fit", "sleeve", "color", "pattern", "type", "generic name",
    "occasion", "neck", "neck type", "closure",
}
# Yes/No attributes: the query word matches the attribute NAME
# (e.g. "hooded" -> attribute "Hooded": Yes / No).
BOOLEAN_KEYS = {"hooded", "reversible"}
# Gender words (after preprocessing) handled by the gender signal only.
GENDER_TERMS = {"men", "woman", "boy", "girl", "unisex"}

FEEDBACK_DEPTH = 20     # top-k BM25 results used to infer the category
MIN_SUPPORT = 20        # a word must be a value of an attribute in >= 20
                        # products to count as an attribute word

# Evidence factors (all in (0, 1]).
CATEGORY_FLOOR = 0.5    # K = 0.5 + 0.5 * share(category) / max share
GENDER_FACTOR = {"match": 1.0, "unknown": 0.7, "mismatch": 0.3}
ATTRIBUTE_FLOOR = 0.5   # A = 0.5 + 0.5 * agreement ratio


@lru_cache(maxsize=None)
def _tokens(text: str) -> frozenset:
    """Processed tokens of an attribute value (cached: values repeat a lot)."""
    return frozenset(preprocess_text(text))


def iter_attributes(details):
    """Yield ``(key, value)`` pairs from a raw product_details field.

    Args:
        details: List of one-item dicts, a dict, or None/NaN.

    Yields:
        ``(lowercased key, value as string)``.
    """
    if isinstance(details, dict):
        details = [details]
    if not isinstance(details, list):
        return
    for item in details:
        if isinstance(item, dict):
            for key, value in item.items():
                if key is not None and value is not None:
                    yield str(key).strip().lower(), str(value)


def product_genders(attributes: dict, title_tokens) -> frozenset:
    """Audience of a product: from "Ideal For" if present, else the title.

    Args:
        attributes: ``{key: frozenset of value tokens}`` of the product.
        title_tokens: Processed title tokens.

    Returns:
        Subset of {"men", "woman", "boy", "girl"}; empty if unknown.
    """
    tokens = attributes.get("ideal for") or frozenset(title_tokens)
    genders = set(tokens) & {"men", "woman", "boy", "girl"}
    if "unisex" in tokens:
        genders |= {"men", "woman"}
    return frozenset(genders)


class IntentScorer:
    """Precomputes product attributes and genders, then scores queries.

    Args:
        index: The shared search index.
        products: DataFrame from ``load_products`` (indexed by pid), with the
            raw ``product_details``, ``title_tokens``, ``sub_category`` and
            ``out_of_stock`` columns.
    """

    def __init__(self, index: SearchIndex, products: pd.DataFrame):
        self.index = index
        self.products = products
        self.attributes = {}
        self.genders = {}
        vocabulary = defaultdict(Counter)  # key -> value token -> #products

        for pid, details, title_tokens in zip(
            products["pid"], products["product_details"],
            products["title_tokens"],
        ):
            attrs = {}
            for key, value in iter_attributes(details):
                if key in ATTRIBUTE_KEYS or key in BOOLEAN_KEYS or key == "ideal for":
                    attrs[key] = attrs.get(key, frozenset()) | _tokens(value)
            self.attributes[pid] = attrs
            self.genders[pid] = product_genders(attrs, title_tokens)
            for key in ATTRIBUTE_KEYS & attrs.keys():
                vocabulary[key].update(attrs[key])

        # Which attributes can each word be a value of? A word that is the
        # attribute's own name ("fit" for Fit) is not treated as a value.
        self.term_keys = defaultdict(set)
        for key, counts in vocabulary.items():
            name_tokens = _tokens(key)
            for term, n in counts.items():
                if n >= MIN_SUPPORT and term not in name_tokens:
                    self.term_keys[term].add(key)
        self.boolean_names = {key: _tokens(key) for key in BOOLEAN_KEYS}

    # -- query understanding --------------------------------------------------

    def query_intent(self, query: str) -> dict:
        """Infer gender and attribute constraints from the query text.

        Args:
            query: Raw query text.

        Returns:
            Dict with ``terms`` (processed), ``genders`` (wanted audiences),
            ``attribute_terms`` ({term: set of attribute keys}) and
            ``boolean_terms`` ({term: boolean key}).
        """
        terms = preprocess_text(query)
        genders = set(terms) & {"men", "woman", "boy", "girl"}
        attribute_terms, boolean_terms = {}, {}
        for term in set(terms) - GENDER_TERMS:
            for key, name in self.boolean_names.items():
                if term in name:
                    boolean_terms[term] = key
            if term in self.term_keys and term not in boolean_terms:
                attribute_terms[term] = self.term_keys[term]
        return {"terms": terms, "genders": genders,
                "attribute_terms": attribute_terms,
                "boolean_terms": boolean_terms}

    # -- the three signals ------------------------------------------------------

    def gender_status(self, pid: str, wanted: set) -> str:
        """"match", "mismatch" or "unknown" for one product."""
        if not wanted:
            return "match"  # gender-neutral query: nothing to check
        have = self.genders[pid]
        if not have:
            return "unknown"
        return "match" if have & wanted else "mismatch"

    def attribute_agreement(self, pid: str, intent: dict) -> tuple[int, int, int]:
        """Count agreeing, contradicting and unknown attribute checks.

        Args:
            pid: Product ID.
            intent: Output of :meth:`query_intent`.

        Returns:
            ``(agree, contradict, unknown)``.
        """
        attrs = self.attributes[pid]
        agree = contradict = unknown = 0
        for term, keys in intent["attribute_terms"].items():
            present = [k for k in keys if k in attrs]
            if not present:
                unknown += 1
            elif any(term in attrs[k] for k in present):
                agree += 1
            else:
                contradict += 1
        for term, key in intent["boolean_terms"].items():
            if key not in attrs:
                unknown += 1
            elif "yes" in attrs[key]:
                agree += 1
            else:
                contradict += 1
        return agree, contradict, unknown

    # -- scoring ---------------------------------------------------------------

    def components(self, query: str) -> pd.DataFrame:
        """Every factor of the intent score for each AND candidate.

        Args:
            query: Raw query text.

        Returns:
            DataFrame indexed by pid with the factors T, K, G, A, S, the
            final ``score`` and explanation columns (empty if no candidates).
        """
        bm25 = bm25_scores(query, self.index)
        if not bm25:
            return pd.DataFrame()
        intent = self.query_intent(query)
        best = max(bm25.values()) or 1.0

        # Pseudo-relevance feedback: category distribution of the top BM25.
        top = sorted(bm25, key=lambda pid: (-bm25[pid], pid))[:FEEDBACK_DEPTH]
        shares = Counter(self.products.loc[top, "sub_category"])
        top_share = max(shares.values())

        rows = {}
        for pid, raw in bm25.items():
            product = self.products.loc[pid]
            share = shares.get(product["sub_category"], 0) / top_share
            gender = self.gender_status(pid, intent["genders"])
            agree, contradict, unknown = self.attribute_agreement(pid, intent)
            checks = agree + contradict + unknown
            ratio = (agree + 0.5 * unknown) / checks if checks else 1.0
            factors = {
                "T": raw / best,
                "K": CATEGORY_FLOOR + (1 - CATEGORY_FLOOR) * share,
                "G": GENDER_FACTOR[gender],
                "A": ATTRIBUTE_FLOOR + (1 - ATTRIBUTE_FLOOR) * ratio,
                "S": OUT_OF_STOCK_FACTOR if product["out_of_stock"] else 1.0,
            }
            rows[pid] = {
                **factors,
                "score": factors["T"] * factors["K"] * factors["G"]
                         * factors["A"] * factors["S"],
                "sub_category": product["sub_category"],
                "gender": gender,
                "attr_agree": agree,
                "attr_contradict": contradict,
                "attr_unknown": unknown,
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    def rank(self, query: str, top_k: int | None = None):
        """Rank the AND results with the intent-aware score.

        Args:
            query: Raw query text.
            top_k: Number of results to return (``None`` = all).

        Returns:
            List of ``(pid, score)`` sorted by descending score, ties by pid.
        """
        table = self.components(query)
        if table.empty:
            return []
        results = [(pid, float(s)) for pid, s in table["score"].items()]
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:top_k] if top_k is not None else results
"""Part 2: TF-IDF ranking for conjunctive search results."""

import math
from collections import Counter

from project_progress.part_1.preprocessing import preprocess_text
from project_progress.part_2.indexing import (
    load_processed_documents,
    build_inverted_index,
    search_and,
)


def rank_tfidf(query, documents, inverted_index, top_k=None):
    """Rank matching documents using TF-IDF cosine similarity."""

    # Step 1: Find documents matching ALL query terms
    candidate_pids = search_and(query, inverted_index)

    if not candidate_pids:
        return []

    # Step 2: Calculate IDF for each term
    n_docs = len(documents)

    idf = {
        term: math.log((n_docs + 1) / (len(pids) + 1)) + 1
        for term, pids in inverted_index.items()
    }

    # Step 3: Calculate query TF-IDF weights
    query_tf = Counter(preprocess_text(query))

    query_weights = {
        term: (1 + math.log(freq)) * idf[term]
        for term, freq in query_tf.items()
    }

    query_norm = math.sqrt(
        sum(weight ** 2 for weight in query_weights.values())
    )

    # Step 4: Calculate document TF-IDF weights
    ranked_results = []

    for pid in candidate_pids:
        term_counts = Counter(documents[pid])

        doc_weights = {
            term: (1 + math.log(freq)) * idf[term]
            for term, freq in term_counts.items()
        }

        doc_norm = math.sqrt(
            sum(weight ** 2 for weight in doc_weights.values())
        )

        # Step 5: Cosine similarity
        dot_product = sum(
            query_weight * doc_weights.get(term, 0)
            for term, query_weight in query_weights.items()
        )

        score = (
            dot_product / (query_norm * doc_norm)
            if query_norm > 0 and doc_norm > 0
            else 0.0
        )

        ranked_results.append((pid, score))

    # Step 6: Sort by descending relevance score
    ranked_results.sort(key=lambda x: (-x[1], x[0]))

    if top_k is not None:
        return ranked_results[:top_k]

    return ranked_results


if __name__ == "__main__":

    documents = load_processed_documents()
    inverted_index = build_inverted_index(documents)

    query = "blue jeans"

    results = rank_tfidf(query, documents, inverted_index, top_k=5)

    print("Query:", query)
    print("\nTop 5 ranked products:")

    for position, (pid, score) in enumerate(results, start=1):
        print(f"{position}. Product: {pid} | TF-IDF score: {score:.4f}")
        
   
    # Validate TF-IDF ranking
    assert len(results) == 5

    assert all(
        0 <= score <= 1
        for pid, score in results
    )

    assert all(
        results[i][1] >= results[i + 1][1]
        for i in range(len(results) - 1)
    )

    assert len(set(pid for pid, score in results)) == len(results)

    print("\nTF-IDF ranking validation passed!")
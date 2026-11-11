"""Part 3: word2vec + cosine ranking (and Doc2Vec, for question 3).

word2vec gives one vector per word. As the assignment specifies, a text
(product or query) is represented by the AVERAGE of the vectors of its
words, and the AND candidates are ranked by cosine similarity between the
query vector and each product vector.

We train word2vec on our own corpus (the processed ``search_tokens``),
not a pre-trained model, because:
  * the tokens are stemmed ("sleev", "kurta"), so a general-English model
    such as Google News would miss most of them;
  * fashion vocabulary ("kurta", "churidar", "jogger") is domain-specific.

Training is made reproducible with a fixed seed and a single worker.
"""

from pathlib import Path

import numpy as np

from project_progress.part_1.preprocessing import preprocess_text
from project_progress.part_3.corpus import SearchIndex


MODELS_DIR = Path(__file__).resolve().parent / "models"

W2V_PARAMS = {
    "vector_size": 100,  # dimensions of each word vector
    "window": 5,         # context words on each side
    "min_count": 2,      # ignore words seen only once (typos, codes)
    "sg": 1,             # skip-gram: better for rare words than CBOW
    "epochs": 10,
    "seed": 42,
    "workers": 1,        # one thread -> reproducible vectors
}

D2V_PARAMS = {
    "vector_size": 100,
    "window": 5,
    "min_count": 2,
    "dm": 0,              # PV-DBOW
    "dbow_words": 1,      # also learn word vectors (helps short texts)
    "epochs": 20,
    "seed": 42,
    "workers": 1,
}


# ---------------------------------------------------------------------------
# Training (requires gensim)
# ---------------------------------------------------------------------------

def train_word2vec(documents: dict[str, list[str]], params: dict = W2V_PARAMS):
    """Train a word2vec model on the processed product texts.

    Args:
        documents: ``{pid: tokens}``; each product is one training sentence.
        params: Keyword arguments for ``gensim.models.Word2Vec``.

    Returns:
        The trained ``gensim.models.Word2Vec`` model.
    """
    from gensim.models import Word2Vec

    sentences = [tokens for tokens in documents.values() if tokens]
    return Word2Vec(sentences=sentences, **params)


def train_doc2vec(documents: dict[str, list[str]], params: dict = D2V_PARAMS):
    """Train a Doc2Vec (PV-DBOW) model, one tagged document per product.

    Args:
        documents: ``{pid: tokens}``.
        params: Keyword arguments for ``gensim.models.Doc2Vec``.

    Returns:
        The trained ``gensim.models.Doc2Vec`` model.
    """
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    corpus = [
        TaggedDocument(words=tokens, tags=[pid])
        for pid, tokens in documents.items() if tokens
    ]
    return Doc2Vec(documents=corpus, **params)


def load_or_train_word2vec(documents: dict[str, list[str]], retrain=False):
    """Load the saved word2vec model, or train and save it.

    Args:
        documents: ``{pid: tokens}``.
        retrain: Force training even if a saved model exists.

    Returns:
        The word vectors (``gensim.models.KeyedVectors``).
    """
    from gensim.models import Word2Vec

    path = MODELS_DIR / "word2vec.model"
    if path.exists() and not retrain:
        return Word2Vec.load(str(path)).wv
    MODELS_DIR.mkdir(exist_ok=True)
    model = train_word2vec(documents)
    model.save(str(path))
    return model.wv


def load_or_train_doc2vec(documents: dict[str, list[str]], retrain=False):
    """Load the saved Doc2Vec model, or train and save it.

    Args:
        documents: ``{pid: tokens}``.
        retrain: Force training even if a saved model exists.

    Returns:
        The trained ``gensim.models.Doc2Vec`` model.
    """
    from gensim.models.doc2vec import Doc2Vec

    path = MODELS_DIR / "doc2vec.model"
    if path.exists() and not retrain:
        return Doc2Vec.load(str(path))
    MODELS_DIR.mkdir(exist_ok=True)
    model = train_doc2vec(documents)
    model.save(str(path))
    return model


# ---------------------------------------------------------------------------
# Text vectors and cosine ranking (work with any {word: vector} lookup)
# ---------------------------------------------------------------------------

def text_vector(tokens: list[str], word_vectors) -> np.ndarray | None:
    """Average the vectors of the tokens that the model knows.

    Repeated tokens are counted each time, as in the assignment's example
    (text vector = (v1 + ... + vn) / n). Unknown tokens are skipped.

    Args:
        tokens: Processed tokens of a product or query.
        word_vectors: Anything supporting ``word in word_vectors`` and
            ``word_vectors[word]`` (gensim KeyedVectors or a dict).

    Returns:
        The mean vector, or ``None`` if no token has a vector.
    """
    vectors = [word_vectors[t] for t in tokens if t in word_vectors]
    if not vectors:
        return None
    return np.mean(np.asarray(vectors, dtype=float), axis=0)


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity of two vectors (0.0 if either is all zeros).

    Args:
        u: First vector.
        v: Second vector.

    Returns:
        A value in [-1, 1].
    """
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.dot(u, v) / norm) if norm else 0.0


def rank_by_vectors(candidates, query_vec, doc_vector, top_k=None):
    """Rank candidates by cosine(query vector, document vector).

    Args:
        candidates: pids to rank (the AND results).
        query_vec: Query vector, or None.
        doc_vector: Function ``pid -> vector or None``.
        top_k: Number of results to return (``None`` = all).

    Returns:
        List of ``(pid, score)``; products without a vector score 0.0.
    """
    if query_vec is None:
        scored = [(pid, 0.0) for pid in candidates]
    else:
        scored = []
        for pid in candidates:
            vec = doc_vector(pid)
            scored.append((pid, cosine(query_vec, vec) if vec is not None else 0.0))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:top_k] if top_k is not None else scored


def rank_word2vec(
    query: str, index: SearchIndex, word_vectors, top_k: int | None = None,
    doc_vectors: dict | None = None,
):
    """Rank the AND results by word2vec (averaged) + cosine similarity.

    Args:
        query: Raw query text.
        index: The shared search index.
        word_vectors: gensim KeyedVectors (or any ``{word: vector}``).
        top_k: Number of results to return (``None`` = all).
        doc_vectors: Optional cache ``{pid: vector}`` to avoid recomputing.

    Returns:
        List of ``(pid, score)``.
    """
    candidates = index.candidates(query)
    if not candidates:
        return []
    query_vec = text_vector(preprocess_text(query), word_vectors)

    def doc_vector(pid):
        if doc_vectors is not None and pid in doc_vectors:
            return doc_vectors[pid]
        vec = text_vector(index.documents[pid], word_vectors)
        if doc_vectors is not None:
            doc_vectors[pid] = vec
        return vec

    return rank_by_vectors(candidates, query_vec, doc_vector, top_k)


def rank_doc2vec(query: str, index: SearchIndex, model, top_k: int | None = None):
    """Rank the AND results by Doc2Vec + cosine similarity.

    Product vectors are the ones learned during training; the query vector
    is inferred with ``model.infer_vector`` (fixed seed for repeatability).

    Args:
        query: Raw query text.
        index: The shared search index.
        model: A trained gensim Doc2Vec model.
        top_k: Number of results to return (``None`` = all).

    Returns:
        List of ``(pid, score)``.
    """
    candidates = index.candidates(query)
    if not candidates:
        return []
    model.random.seed(42)
    query_vec = model.infer_vector(preprocess_text(query), epochs=100)

    def doc_vector(pid):
        return model.dv[pid] if pid in model.dv else None

    return rank_by_vectors(candidates, query_vec, doc_vector, top_k)
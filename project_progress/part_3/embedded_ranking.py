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

Training is made reproducible with a fixed seed, a single worker and a
stable hash function (gensim uses Python's ``hash`` by default, which is
randomised in every new process unless PYTHONHASHSEED is set).
"""

import hashlib
import json
import math
import zlib
from pathlib import Path

import numpy as np

from project_progress.part_1.preprocessing import preprocess_text
from project_progress.part_3.corpus import SearchIndex, check_top_k


MODELS_DIR = Path(__file__).resolve().parent / "models"


def stable_hash(text: str) -> int:
    """Hash that is the same in every Python process (unlike ``hash``)."""
    return zlib.crc32(text.encode("utf-8"))


W2V_PARAMS = {
    "vector_size": 100,  # dimensions of each word vector
    "window": 5,         # context words on each side
    "min_count": 2,      # ignore words seen only once (typos, codes)
    "sg": 1,             # skip-gram: better for rare words than CBOW
    "epochs": 10,
    "seed": 42,
    "workers": 1,        # one thread -> reproducible vectors
    "hashfxn": stable_hash,  # same initial vectors in every process
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
    "hashfxn": stable_hash,
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


def fingerprint(documents: dict[str, list[str]], params: dict) -> str:
    """SHA-256 of the training corpus and parameters.

    Saved next to each model, so a model trained on other data or with
    other parameters is detected as stale and retrained. The documents are
    hashed in the order in which they are passed, because that is also the
    order in which they are trained on, and training order changes the
    model. Same fingerprint therefore means same training input.
    """
    digest = hashlib.sha256()
    settings = {k: v.__name__ if callable(v) else v for k, v in params.items()}
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for pid in documents:
        digest.update(pid.encode("utf-8"))
        digest.update(b"\0" + " ".join(documents[pid]).encode("utf-8") + b"\n")
    return digest.hexdigest()


def _is_fresh(path: Path, expected: str) -> bool:
    """True if ``path`` exists and was trained on the expected fingerprint."""
    stamp = path.with_suffix(".fingerprint")
    return (path.exists() and stamp.exists()
            and stamp.read_text().strip() == expected)


def load_or_train_word2vec(documents: dict[str, list[str]], retrain=False):
    """Load the saved word2vec model, or train and save it.

    The saved model is reused only if its fingerprint (corpus + parameters)
    matches; otherwise it is retrained.

    Args:
        documents: ``{pid: tokens}``.
        retrain: Force training even if a saved model exists.

    Returns:
        The word vectors (``gensim.models.KeyedVectors``).
    """
    from gensim.models import Word2Vec

    path = MODELS_DIR / "word2vec.model"
    expected = fingerprint(documents, W2V_PARAMS)
    if _is_fresh(path, expected) and not retrain:
        return Word2Vec.load(str(path)).wv
    MODELS_DIR.mkdir(exist_ok=True)
    model = train_word2vec(documents)
    model.save(str(path))
    path.with_suffix(".fingerprint").write_text(expected + "\n")
    return model.wv


def load_or_train_doc2vec(documents: dict[str, list[str]], retrain=False):
    """Load the saved Doc2Vec model, or train and save it.

    As for word2vec, a model whose fingerprint does not match is retrained.

    Args:
        documents: ``{pid: tokens}``.
        retrain: Force training even if a saved model exists.

    Returns:
        The trained ``gensim.models.Doc2Vec`` model.
    """
    from gensim.models.doc2vec import Doc2Vec

    path = MODELS_DIR / "doc2vec.model"
    expected = fingerprint(documents, D2V_PARAMS)
    if _is_fresh(path, expected) and not retrain:
        return Doc2Vec.load(str(path))
    MODELS_DIR.mkdir(exist_ok=True)
    model = train_doc2vec(documents)
    model.save(str(path))
    path.with_suffix(".fingerprint").write_text(expected + "\n")
    return model


def infer_query_vector(model, tokens: list[str], epochs: int = 100) -> np.ndarray:
    """Infer a PV-DBOW query vector that is identical in every Python process.

    gensim's ``Doc2Vec.infer_vector`` starts from a pseudo-random vector
    seeded with Python's built-in ``hash`` of the text, which is randomised
    in every new process, so the inferred vector changed between runs.

    This function runs the same algorithm as ``infer_vector`` (same initial
    vector formula, same linearly decaying learning rate, same gensim
    training routine), but builds the initial vector with
    :func:`stable_hash`. Unlike swapping gensim's module-level function, it
    changes no global state, so it is safe to call from concurrent requests
    (e.g. a web server) as long as each thread uses its own model copy or the
    calls do not share ``model.random``.

    Args:
        model: A trained gensim Doc2Vec model in PV-DBOW mode (``dm=0``).
        tokens: Processed query tokens.
        epochs: Inference epochs.

    Returns:
        The inferred vector.
    """
    from gensim.models.doc2vec import train_document_dbow
    from gensim.models.keyedvectors import pseudorandom_weak_vector

    if not model.sg:
        raise ValueError("infer_query_vector supports PV-DBOW models (dm=0) only")
    size = model.dv.vector_size
    doctag_vectors = pseudorandom_weak_vector(
        size, seed_string=" ".join(tokens), hashfxn=stable_hash
    ).reshape(1, size)
    doctags_lockf = np.ones(1, dtype=np.float32)
    work = np.zeros(model.layer1_size, dtype=np.float32)
    alpha, min_alpha = model.alpha, model.min_alpha
    alpha_delta = (alpha - min_alpha) / max(epochs - 1, 1)
    model.random.seed(42)  # negative sampling draws from model.random
    for _ in range(epochs):
        train_document_dbow(model, tokens, [0], alpha, work,
                            learn_words=False, learn_hidden=False,
                            doctag_vectors=doctag_vectors,
                            doctags_lockf=doctags_lockf)
        alpha -= alpha_delta
    return doctag_vectors[0]


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


def _vector_order(item: tuple[str, float]):
    """Sort key: scored products by descending cosine, then NaN, ties by pid."""
    pid, score = item
    missing = math.isnan(score)
    return (missing, 0.0 if missing else -score, pid)


def rank_by_vectors(candidates, query_vec, doc_vector, top_k=None):
    """Rank candidates by cosine(query vector, document vector).

    Args:
        candidates: pids to rank (the AND results).
        query_vec: Query vector, or None.
        doc_vector: Function ``pid -> vector or None``.
        top_k: Number of results to return (``None`` = all).

    Returns:
        List of ``(pid, score)``. Products without a vector get score NaN
        and are placed AFTER every product that has one (a cosine can be
        negative, so giving them 0.0 could rank them above real matches).
        If the query has no vector, all candidates get NaN (pid order).
    """
    check_top_k(top_k)
    scored = []
    for pid in candidates:
        vec = doc_vector(pid) if query_vec is not None else None
        scored.append((pid, cosine(query_vec, vec) if vec is not None else float("nan")))
    scored.sort(key=_vector_order)
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
    is inferred with :func:`infer_query_vector` (same vector in every run).

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
    query_vec = infer_query_vector(model, preprocess_text(query))

    def doc_vector(pid):
        return model.dv[pid] if pid in model.dv else None

    return rank_by_vectors(candidates, query_vec, doc_vector, top_k)
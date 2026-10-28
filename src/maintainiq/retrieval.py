"""Hybrid retrieval: BM25 (exact tokens) + dense similarity, fused with Reciprocal Rank Fusion.

RRF is rank-based, so BM25 scores and cosine similarities never need to share a scale.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from functools import lru_cache
from typing import Callable, Iterable, Sequence

import numpy as np
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

from maintainiq.config import Settings
from maintainiq.schemas import Document, RetrievalResult

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STEMMER = PorterStemmer()
_STOP = frozenset(ENGLISH_STOP_WORDS) | {"exact"}


@lru_cache(maxsize=65536)
def _stem(word: str) -> str:
    return _STEMMER.stem(word) if word.isalpha() and len(word) > 3 else word


def tokenize(text: str) -> list[str]:
    """Content tokens: lowercase, stopwords removed, Porter-stemmed.

    Hyphenated codes such as `brg-wear` are kept whole (exact fault-code match) and also split
    into stemmed parts, so partial queries still match."""
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP:
            continue
        if "-" in tok:
            tokens.append(tok)
            tokens.extend(_stem(part) for part in tok.split("-") if part and part not in _STOP)
        else:
            tokens.append(_stem(tok))
    return tokens


class BM25Index:
    def __init__(self, texts: Sequence[str]):
        self._bm25 = BM25Okapi([tokenize(t) for t in texts]) if texts else None

    def scores(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            return np.zeros(0)
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)


class DenseIndex:
    """Dense vectors. Default backend is TF-IDF + LSA, which runs offline with no model download.
    The sentence-transformers backend is used when EMBEDDING_BACKEND=sentence-transformers."""

    def __init__(self, texts: Sequence[str], settings: Settings):
        self._encode: Callable[[list[str]], np.ndarray]
        if settings.embedding_backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer  # heavy import, optional

            model = SentenceTransformer(settings.embedding_model)
            self._encode = lambda xs: np.asarray(model.encode(xs, normalize_embeddings=True))
            self.backend = f"sentence-transformers:{settings.embedding_model}"
            self._matrix = self._encode(list(texts)) if texts else np.zeros((0, 1))
        else:
            self._fit_lsa(texts)
            self.backend = "tfidf-lsa"

    def _fit_lsa(self, texts: Sequence[str]) -> None:
        if not texts:
            self._vectorizer, self._svd, self._matrix = None, None, np.zeros((0, 1))
            self._encode = lambda xs: np.zeros((len(xs), 1))
            return
        self._vectorizer = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True, ngram_range=(1, 2)
        )
        tfidf = self._vectorizer.fit_transform(texts)
        n_components = max(1, min(128, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=0)
        self._matrix = normalize(self._svd.fit_transform(tfidf))
        self._encode = lambda xs: normalize(self._svd.transform(self._vectorizer.transform(xs)))

    def similarities(self, query: str) -> np.ndarray:
        if self._matrix.shape[0] == 0:
            return np.zeros(0)
        q = self._encode([query])[0]
        return self._matrix @ q


class HybridRetriever:
    def __init__(self, documents: Sequence[Document], settings: Settings):
        if len({d.doc_id for d in documents}) != len(documents):
            raise ValueError("document ids must be unique")
        self.documents = list(documents)
        self.settings = settings
        texts = [f"{d.title}. {d.text}" for d in self.documents]
        self._bm25 = BM25Index(texts)
        self._dense = DenseIndex(texts, settings)
        logger.info("indexed %d documents (dense backend: %s)", len(texts), self._dense.backend)

    def _allowed(self, filters: dict | None) -> np.ndarray:
        """Boolean mask. A filter value may be one value or a set of allowed values.

        Documents that do not carry the filtered key are treated as shared (for example, manuals
        apply to every motor), so an equipment-scoped query still sees the relevant manuals."""
        mask = np.ones(len(self.documents), dtype=bool)
        for key, value in (filters or {}).items():
            allowed = set(value) if isinstance(value, (set, frozenset, list, tuple)) else {value}
            for i, doc in enumerate(self.documents):
                actual = doc.source_type if key == "source_type" else doc.metadata.get(key)
                if actual is None and key != "source_type":
                    continue  # shared document
                if actual not in allowed:
                    mask[i] = False
        return mask

    def search(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[RetrievalResult]:
        if not query.strip() or not self.documents:
            return []
        mask = self._allowed(filters)
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return []

        bm25 = self._bm25.scores(query)
        dense = self._dense.similarities(query)

        bm25_ranked = _rank(candidates, bm25, self.settings.candidate_k, positive_only=True)
        dense_ranked = _rank(candidates, dense, self.settings.candidate_k, positive_only=False)
        return self._fuse(bm25_ranked, dense_ranked, dense, top_k)

    def _fuse(self, bm25_ranked: list[int], dense_ranked: list[int], dense: np.ndarray,
              top_k: int) -> list[RetrievalResult]:
        k = self.settings.rrf_k
        fused: dict[int, float] = defaultdict(float)
        bm25_pos = {i: r for r, i in enumerate(bm25_ranked, start=1)}
        dense_pos = {i: r for r, i in enumerate(dense_ranked, start=1)}
        for ranking in (bm25_ranked, dense_ranked):
            for rank, idx in enumerate(ranking, start=1):
                fused[idx] += 1.0 / (k + rank)
        order = sorted(fused, key=lambda i: (-fused[i], self.documents[i].doc_id))[:top_k]
        return [
            RetrievalResult(
                document=self.documents[i],
                rrf_score=fused[i],
                bm25_rank=bm25_pos.get(i),
                dense_rank=dense_pos.get(i),
                dense_similarity=float(dense[i]) if dense.size else 0.0,
            )
            for i in order
        ]


def _rank(candidates: np.ndarray, scores: np.ndarray, limit: int, positive_only: bool) -> list[int]:
    if scores.size == 0:
        return []
    ordered = sorted(candidates.tolist(), key=lambda i: (-scores[i], i))
    if positive_only:
        ordered = [i for i in ordered if scores[i] > 0]
    return ordered[:limit]


def rrf_fuse(rankings: Iterable[Sequence[str]], k: int = 60) -> list[str]:
    """Standalone RRF over ranked id lists. Used by tests and the evaluation harness."""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] += 1.0 / (k + rank)
    return sorted(fused, key=lambda i: (-fused[i], i))

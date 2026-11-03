"""Hybrid retrieval: BM25 + vector search, fused with Reciprocal Rank Fusion,
optionally reranked with a cross-encoder.

Why hybrid: dense embeddings are good at meaning ("who leads the scoring
charts") but weak on exact tokens such as names, numbers and rare words
("Saïss", "48 teams"). BM25 is the opposite. Reciprocal Rank Fusion combines
the two rankings without having to calibrate their very different scores.

This module has no LangChain imports, so the ranking logic can be unit-tested
without downloading any model. Chunks only need `.page_content` and `.metadata`.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# Small English stopword list: enough to stop "the/of/is" from dominating BM25
# on short questions, without pulling in NLTK.
STOPWORDS = frozenset("""
a an and are as at be by did do does for from has have how in is it its of on or
that the their this to was were what when where which who whom why will with
""".split())

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    """Okapi BM25 over a fixed list of texts (pure Python)."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {term: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for term, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = [t for t in tokenize(query) if t in self.idf]
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len) if self.avg_len else self.k1
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            out.append(s)
        return out

    def top_n(self, query: str, n: int) -> list[int]:
        """Indices of the n best-scoring texts (only those with a score above 0)."""
        scored = [(s, i) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda p: (-p[0], p[1]))
        return [i for _, i in scored[:n]]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Fuse several rankings of item ids: score(d) = sum over rankings of 1 / (k + rank).

    k = 60 is the constant from Cormack et al. (2009); it damps the influence of
    the very top ranks so that one retriever cannot dominate.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=lambda item: (-fused[item], item))


def _key(doc) -> tuple:
    # Content + source identifies a chunk; other metadata (page labels, PDF producer...)
    # is not needed and may not survive the JSON round-trip of the chunk cache.
    return (doc.page_content, str((doc.metadata or {}).get("source", "")))


MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


class HybridRetriever:
    """Drop-in replacement for a LangChain retriever: `invoke(query) -> list of chunks`.

    mode:
      vector         FAISS similarity search only (the original pipeline)
      bm25           keyword search only
      hybrid         BM25 + vector, fused with RRF
      hybrid_rerank  hybrid candidates, then reordered by a cross-encoder
    """

    def __init__(self, vectorstore, chunks, mode: str = "hybrid_rerank", k: int = 3,
                 candidates: int = 20, reranker=None, rrf_k: int = 60):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "hybrid_rerank" and reranker is None:
            raise ValueError("mode 'hybrid_rerank' needs a reranker")
        self.vectorstore, self.chunks = vectorstore, chunks
        self.mode, self.k, self.candidates, self.rrf_k = mode, k, candidates, rrf_k
        self.reranker = reranker
        self.bm25 = BM25Index([c.page_content for c in chunks]) if mode != "vector" else None
        self._index = {}
        for i, c in enumerate(chunks):
            self._index.setdefault(_key(c), i)

    def _vector_ids(self, query: str, n: int) -> list[int]:
        ids = []
        for doc in self.vectorstore.similarity_search(query, k=n):
            i = self._index.get(_key(doc))
            if i is not None and i not in ids:
                ids.append(i)
        return ids

    def invoke(self, query: str) -> list:
        if self.mode == "vector":
            return [self.chunks[i] for i in self._vector_ids(query, self.k)]
        if self.mode == "bm25":
            return [self.chunks[i] for i in self.bm25.top_n(query, self.k)]

        fused = reciprocal_rank_fusion(
            [self._vector_ids(query, self.candidates), self.bm25.top_n(query, self.candidates)],
            k=self.rrf_k,
        )
        if self.mode == "hybrid":
            return [self.chunks[i] for i in fused[: self.k]]

        pool = fused[: self.candidates]
        scores = self.reranker.score(query, [self.chunks[i].page_content for i in pool])
        order = sorted(range(len(pool)), key=lambda j: -scores[j])
        return [self.chunks[pool[j]] for j in order[: self.k]]


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly, which is more accurate than comparing
    two separately computed embeddings, but too slow to run over the whole corpus.
    That is why it only reorders the fused top candidates."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # heavy import, only when used
        self.model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self.model.predict([(query, p) for p in passages])]


# ----------------------------------------------------------------------------- evaluation helpers
def is_relevant(doc, expected_sources: list[str]) -> bool:
    source = str((doc.metadata or {}).get("source", ""))
    return any(e in source for e in expected_sources)


def hit_at(docs: list, expected_sources: list[str], k: int) -> float:
    """1.0 if any of the first k chunks comes from an expected source."""
    return float(any(is_relevant(d, expected_sources) for d in docs[:k]))


def reciprocal_rank(docs: list, expected_sources: list[str]) -> float:
    """1 / rank of the first chunk from an expected source (0 if none)."""
    for rank, d in enumerate(docs, start=1):
        if is_relevant(d, expected_sources):
            return 1.0 / rank
    return 0.0

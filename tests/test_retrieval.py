"""Unit tests for retrieval.py. No models or network needed:

    python -m unittest discover tests
"""
import os
import sys
import unittest
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import (BM25Index, HybridRetriever, hit_at, reciprocal_rank,  # noqa: E402
                       reciprocal_rank_fusion, tokenize)


@dataclass
class Doc:
    page_content: str
    metadata: dict = field(default_factory=dict)


CHUNKS = [
    Doc("Romain Saïss captained Morocco at the 2022 World Cup in Qatar.", {"source": "wiki/Morocco"}),
    Doc("The 2026 World Cup is the first edition with 48 teams.", {"source": "wiki/2026_FIFA_World_Cup"}),
    Doc("Lionel Messi is an Argentine footballer who plays as a forward.", {"source": "wiki/Lionel_Messi"}),
    Doc("Canada, Mexico and the United States co-host the tournament.", {"source": "wiki/2026_FIFA_World_Cup"}),
    Doc("The Ballon d'Or is an annual award for the best footballer.", {"source": "wiki/Ballon_d'Or"}),
]


class FakeVectorStore:
    """Returns chunks in a fixed order, like a FAISS index would return by similarity."""

    def __init__(self, order):
        self.order = order

    def similarity_search(self, query, k):
        return [Doc(CHUNKS[i].page_content, dict(CHUNKS[i].metadata)) for i in self.order[:k]]


class ReverseReranker:
    """Prefers shorter passages, just to prove the reranker controls the final order."""

    def score(self, query, passages):
        return [-len(p) for p in passages]


class TokenizeTest(unittest.TestCase):
    def test_lowercases_keeps_accents_and_drops_stopwords(self):
        self.assertEqual(tokenize("Who is the captain of Saïss?"), ["captain", "saïss"])


class BM25Test(unittest.TestCase):
    def setUp(self):
        self.bm25 = BM25Index([c.page_content for c in CHUNKS])

    def test_exact_name_ranks_first(self):
        self.assertEqual(self.bm25.top_n("Saïss captain", 1), [0])

    def test_number_match(self):
        self.assertEqual(self.bm25.top_n("how many teams 48", 1), [1])

    def test_unknown_terms_return_nothing(self):
        self.assertEqual(self.bm25.top_n("cricket badminton", 3), [])

    def test_rare_term_outweighs_common_term(self):
        scores = self.bm25.scores("ballon")
        self.assertGreater(scores[4], 0)
        self.assertEqual(sum(1 for s in scores if s > 0), 1)


class FusionTest(unittest.TestCase):
    def test_item_ranked_well_by_both_wins(self):
        # 7 is 2nd in both lists; 1 and 9 are each 1st in only one list.
        fused = reciprocal_rank_fusion([[1, 7, 3], [9, 7, 4]])
        self.assertEqual(fused[0], 7)

    def test_scores_follow_formula(self):
        fused = reciprocal_rank_fusion([[5], [6, 5]], k=60)
        # 5: 1/61 + 1/62 ; 6: 1/61
        self.assertEqual(fused, [5, 6])

    def test_ties_broken_by_id(self):
        self.assertEqual(reciprocal_rank_fusion([[2], [1]]), [1, 2])


class HybridRetrieverTest(unittest.TestCase):
    def test_vector_mode_uses_vector_order(self):
        r = HybridRetriever(FakeVectorStore([2, 0, 1]), CHUNKS, mode="vector", k=2)
        self.assertEqual([d.metadata["source"] for d in r.invoke("x")], ["wiki/Lionel_Messi", "wiki/Morocco"])

    def test_hybrid_recovers_exact_match_missed_by_vectors(self):
        # Vector search ranks the Saïss chunk last; BM25 finds it by name.
        r = HybridRetriever(FakeVectorStore([2, 3, 1, 4, 0]), CHUNKS, mode="hybrid", k=2, candidates=5)
        self.assertIn("wiki/Morocco", [d.metadata["source"] for d in r.invoke("Saïss captain Morocco")])

    def test_rerank_controls_final_order(self):
        r = HybridRetriever(FakeVectorStore([0, 1, 2, 3, 4]), CHUNKS, mode="hybrid_rerank", k=1,
                            candidates=5, reranker=ReverseReranker())
        shortest = min(CHUNKS, key=lambda c: len(c.page_content))
        self.assertEqual(r.invoke("world cup footballer")[0].page_content, shortest.page_content)

    def test_rerank_mode_requires_reranker(self):
        with self.assertRaises(ValueError):
            HybridRetriever(FakeVectorStore([]), CHUNKS, mode="hybrid_rerank")


class MetricsTest(unittest.TestCase):
    def test_hit_and_reciprocal_rank(self):
        docs = [CHUNKS[2], CHUNKS[1], CHUNKS[0]]
        self.assertEqual(hit_at(docs, ["2026_FIFA"], 1), 0.0)
        self.assertEqual(hit_at(docs, ["2026_FIFA"], 2), 1.0)
        self.assertEqual(reciprocal_rank(docs, ["2026_FIFA"]), 0.5)
        self.assertEqual(reciprocal_rank(docs, ["nowhere"]), 0.0)


if __name__ == "__main__":
    unittest.main()

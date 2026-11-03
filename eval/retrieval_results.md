# Retrieval results (19 questions, 912 chunks)

| Mode | Hit@1 | Hit@3 | MRR@10 | ms / query |
|---|---|---|---|---|
| vector | 0.95 | 1.00 | 0.974 | 13 |
| bm25 | 0.74 | 0.95 | 0.826 | 2 |
| hybrid | 1.00 | 1.00 | 1.000 | 15 |
| hybrid_rerank | 0.89 | 0.95 | 0.932 | 1616 |

- Missed in top 3 (vector): none
- Missed in top 3 (bm25): morocco_captain
- Missed in top 3 (hybrid): none
- Missed in top 3 (hybrid_rerank): morocco_captain

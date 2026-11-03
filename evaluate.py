"""Compare retrieval modes on the labelled questions in eval/questions.json.

    python evaluate.py

No Groq key is needed: this measures retrieval only (did the right source come
back?), not the LLM's answer. Results are printed and written to
eval/retrieval_results.md.

Metrics, over questions that have an expected source:
  Hit@1   share of questions whose top chunk comes from an expected source
  Hit@3   same, within the top 3 (the pipeline sends 3 chunks to the LLM)
  MRR@10  mean of 1 / rank of the first relevant chunk in the top 10
"""
import json
import logging
import os
import time

from pipeline import load_or_build_index, build_retriever
from retrieval import MODES, CrossEncoderReranker, hit_at, reciprocal_rank

logging.getLogger().setLevel(logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS = os.path.join(HERE, "eval", "questions.json")
OUT = os.path.join(HERE, "eval", "retrieval_results.md")
DEPTH = 10


def main():
    with open(QUESTIONS, encoding="utf-8") as f:
        questions = [q for q in json.load(f)["questions"] if q["expected_sources"]]

    vectorstore, chunks = load_or_build_index()
    reranker = CrossEncoderReranker()

    rows, misses = [], {}
    for mode in MODES:
        retriever = build_retriever(vectorstore, chunks, mode=mode, k=DEPTH, reranker=reranker)
        h1 = h3 = mrr = 0.0
        start = time.perf_counter()
        for q in questions:
            docs = retriever.invoke(q["question"])
            h1 += hit_at(docs, q["expected_sources"], 1)
            h3 += hit_at(docs, q["expected_sources"], 3)
            mrr += reciprocal_rank(docs, q["expected_sources"])
            if not hit_at(docs, q["expected_sources"], 3):
                misses.setdefault(mode, []).append(q["id"])
        ms = 1000 * (time.perf_counter() - start) / len(questions)
        n = len(questions)
        rows.append((mode, h1 / n, h3 / n, mrr / n, ms))

    lines = [
        f"# Retrieval results ({len(questions)} questions, {len(chunks)} chunks)\n",
        "| Mode | Hit@1 | Hit@3 | MRR@10 | ms / query |",
        "|---|---|---|---|---|",
    ]
    for mode, h1, h3, mrr, ms in rows:
        lines.append(f"| {mode} | {h1:.2f} | {h3:.2f} | {mrr:.3f} | {ms:.0f} |")
    lines.append("")
    for mode in MODES:
        lines.append(f"- Missed in top 3 ({mode}): {', '.join(misses.get(mode, [])) or 'none'}")

    report = "\n".join(lines) + "\n"
    print(report)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved to {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()

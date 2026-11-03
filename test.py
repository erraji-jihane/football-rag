"""Answer-level test: runs the original 10 questions through the full pipeline
(retrieval + Groq) and scores each answer against eval/questions.json.

    python test.py

PASS/FAIL comes from simple keyword checks; MANUAL means the correct answer
changes during the tournament, so read it and judge it yourself.
"""
import json
import os

from pipeline import build_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
DECLINE_PHRASES = ("don't know", "do not know", "dont know", "not mentioned", "no information")


def score(answer: str, check: dict) -> str:
    text = answer.lower()
    kind = check["type"]
    if kind == "manual":
        return "MANUAL"
    if kind == "decline":
        return "PASS" if any(p in text for p in DECLINE_PHRASES) else "FAIL"
    hits = [k.lower() in text for k in check["keywords"]]
    ok = all(hits) if kind == "all" else any(hits)
    return "PASS" if ok else "FAIL"


def main():
    with open(os.path.join(HERE, "eval", "questions.json"), encoding="utf-8") as f:
        questions = [q for q in json.load(f)["questions"] if not q["id"].startswith("r_")]

    retriever, llm, prompt = build_pipeline()
    tally = {"PASS": 0, "FAIL": 0, "MANUAL": 0}

    for q in questions:
        docs = retriever.invoke(q["question"])
        context = "\n\n".join(doc.page_content for doc in docs)
        answer = llm.invoke(prompt.format(context=context, question=q["question"])).content
        verdict = score(answer, q["answer_check"])
        tally[verdict] += 1

        print(f"\n[{verdict}] Q: {q['question']}")
        print(f"A: {answer}")
        print(f"Sources: {[doc.metadata.get('source') for doc in docs]}")

    graded = tally["PASS"] + tally["FAIL"]
    print(f"\nAutomatic checks: {tally['PASS']}/{graded} passed; {tally['MANUAL']} to check by hand.")


if __name__ == "__main__":
    main()

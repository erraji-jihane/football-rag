from pipeline import build_pipeline

retriever, llm, prompt = build_pipeline()

test_questions = [
    "who is lionel messi",
    "what club does kylian mbappé play for",
    "who is morocco's captain at the 2026 world cup",
    "which players are in cape verde's world cup squad",
    "who is leading the golden boot race at the 2026 world cup",
    "how many goals has messi scored at the 2026 world cup",
    "how many teams are in the 2026 world cup",
    "where is the 2026 world cup being hosted",
    "why is curaçao's qualification historic",
    "who won the 2026 world cup",
]

for query in test_questions:
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = prompt.format(context=context, question=query)
    answer = llm.invoke(final_prompt)

    print(f"\nQ: {query}")
    print(f"A: {answer.content}")
    print(f"Sources: {[doc.metadata.get('source') for doc in docs]}")
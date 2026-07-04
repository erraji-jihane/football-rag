from pipeline import build_pipeline

retriever, llm, prompt = build_pipeline()

while True:
    query = input("Ask a question (exit to quit)")

    if query == 'exit':
        break

    #search relevant things
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    #generate final prompt and sends to llm
    final_prompt = prompt.format(context=context, question=query)
    answer = llm.invoke(final_prompt)

    print("Answer:", answer.content)
    print("Sources:", [doc.metadata for doc in docs])
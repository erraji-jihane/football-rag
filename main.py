from pipeline import build_pipeline
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
    try:
        answer = llm.invoke(final_prompt)
        answer_text = answer.content
    except:
        logger.error(f"LLM call failed for query '{query}': {e}")
        answer_text = "Sorry, I couldn't reach the model right now. Please try again."


    print("Answer:", answer.content)
    print("Sources:", [doc.metadata for doc in docs])
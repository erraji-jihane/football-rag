import streamlit as st
from main import vectorstore, llm, prompt

st.title("📄 RAG Document Assistant")

query = st.text_input("Ask a question:")

if query:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = prompt.format(context=context, question=query)
    result = llm.invoke(final_prompt)
    st.write(result.content)
import os
import glob
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file. Please add it.")

#Loading the pdfs
print("GROQ API Key loaded successfully!")

pdf_files = glob.glob("data/*.pdf")
all_documents = []
for path in pdf_files:
    loader = PyPDFLoader(path)
    all_documents.extend(loader.load())

#chuncking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(all_documents)

# creating vector store 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

FAISS_INDEX_PATH = "index/faiss_index"

if os.path.exists(FAISS_INDEX_PATH):
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)

# setting up LLM and query
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use only the context below to answer.
If unsure, say "I don't know based on the document."

Context: {context}
Question: {question}
Answer:"""
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
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
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file. Please add it.")

print("GROQ API Key loaded successfully!")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# config
PDF_PATH = "data/rag_paper.pdf"

# loading ans splitting doc
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")

# creating vector store 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("index/faiss_index")
print("✅ FAISS index saved successfully!")

# setting up LLM and query
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

# Test query
query = "What is retrieval augmented generation?"

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke(query)
context = "\n\n".join([doc.page_content for doc in docs])

prompt = f"""Use the following context to answer the question:

Context:
{context}

Question: {query}

Answer:"""

result = llm.invoke(prompt)
print("\n🔹 Question:", query)
print("🔹 Answer:", result.content)
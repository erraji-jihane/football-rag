import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"

# loading and chunking PDF
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
loader = PyPDFLoader(os.path.join(BASE_DIR, "data", "rag_paper.pdf"))
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

# embedding and storing in FAISS
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("index/faiss_index")
print("Index saved.")

# retrieving and answering
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "What is retrieval augmented generation?"
result = chain.invoke(query)
print("\nAnswer:", result)
from sources import WIKIPEDIA_TOPICS, OTHER_URLS
import os
import glob
import requests
import logging
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def scrap_urls(url):
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=header, timeout=10) #gets raw html
        response.raise_for_status()
    except requests.RequestException:
        return ""
    
    soup = BeautifulSoup(response.text,"html.parser") #parses the html into a searchable structure
    #text = soup.get_text()
    content_div = soup.find("div", {"id": "mw-content-text"})
    if content_div:
        text = content_div.get_text()
    else:
        text = soup.get_text()
    return text

def build_pipeline ():

    load_dotenv()

    # Get API key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in .env file. Please add it.")

    logger.info("GROQ API Key loaded successfully!")

    all_documents = []

    #Loading the pdfs
    pdf_files = glob.glob("data/*.pdf")
    logger.info(f"Loaded {len(pdf_files)} PDF(s)")

    for path in pdf_files:
        loader = PyPDFLoader(path)
        all_documents.extend(loader.load())

    #Loading the links 
    wikipedia_urls = [f"https://en.wikipedia.org/wiki/{topic}" for topic in WIKIPEDIA_TOPICS ]

    urls = wikipedia_urls + OTHER_URLS
    logger.info(f"Preparing to scrape {len(urls)} URLs")

    for url in urls :
        page_text = scrap_urls(url)
        page_doc = Document(page_content=page_text, metadata={"source": url}) #Wraps scraped text + its URL into one Document object, same asteh pdf page.
        all_documents.append(page_doc)


    #chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(all_documents)
    logger.info(f"Created {len(chunks)} chunks")


    # creating vector store 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    FAISS_INDEX_PATH = "index/faiss_icndex"

    if os.path.exists(FAISS_INDEX_PATH):
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("Loaded existing FAISS index")
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        logger.info("Built and saved new FAISS index")

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
    return retriever, llm, prompt
from sources import WIKIPEDIA_TOPICS, OTHER_URLS
import os
import glob
import json
import hashlib
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

from retrieval import HybridRetriever, CrossEncoderReranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INDEX_DIR = "index"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")
MANIFEST_PATH = os.path.join(INDEX_DIR, "manifest.json")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieval mode used by main.py and test.py: vector | bm25 | hybrid | hybrid_rerank
DEFAULT_MODE = os.getenv("RETRIEVAL_MODE", "hybrid_rerank")


def scrap_urls(url):
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=header, timeout=10) #gets raw html
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Scrape failed for {url}: {e}")
        return ""

    soup = BeautifulSoup(response.text,"html.parser") #parses the html into a searchable structure
    content_div = soup.find("div", {"id": "mw-content-text"})
    if content_div:
        text = content_div.get_text()
    else:
        text = soup.get_text()
    return text

def list_urls():
    wikipedia_urls = [f"https://en.wikipedia.org/wiki/{topic}" for topic in WIKIPEDIA_TOPICS]
    return wikipedia_urls + OTHER_URLS

def list_pdfs():
    return sorted(glob.glob("data/*.pdf"))

def load_pdfs():

    all_documents = []

    #Loading the pdfs
    pdf_files = list_pdfs()
    logger.info(f"Found {len(pdf_files)} PDF(s)")

    for path in pdf_files:
        try:
            loader = PyPDFLoader(path)
            all_documents.extend(loader.load())
        except Exception as e:
            logger.info(f"Failed to load PDF '{path}' as {e}")

    return all_documents

def load_links(all_documents):
    #Loading the links
    urls = list_urls()
    logger.info(f"Preparing to scrape {len(urls)} URLs")

    for url in urls :
        page_text = scrap_urls(url)
        if not page_text.strip():
            continue
        page_doc = Document(page_content=page_text, metadata={"source": url}) #Wraps scraped text + its URL into one Document object, same as the pdf page.
        all_documents.append(page_doc)

def chunk_documents(all_documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(all_documents)
    return chunks

def sources_fingerprint():
    """Changes whenever the source list, the PDFs or the chunking settings change."""
    pdfs = [(p, os.path.getsize(p)) for p in list_pdfs()]
    payload = json.dumps({"urls": list_urls(), "pdfs": pdfs, "chunk_size": CHUNK_SIZE,
                          "chunk_overlap": CHUNK_OVERLAP, "embedding_model": EMBEDDING_MODEL})
    return hashlib.sha256(payload.encode()).hexdigest()

def save_chunks(chunks):
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump([{"page_content": c.page_content, "metadata": c.metadata} for c in chunks], f)

def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in json.load(f)]

def load_or_build_index():
    """Returns (vectorstore, chunks).

    The FAISS index and the chunk texts are cached in index/. The cache is reused
    only if the sources are unchanged, so editing sources.py or adding a PDF
    triggers a rebuild automatically (no more deleting index/ by hand). BM25 needs
    the chunk texts, which is why they are cached alongside the vectors.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    fingerprint = sources_fingerprint()

    cached = None
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            cached = json.load(f).get("fingerprint")

    if cached == fingerprint and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        chunks = load_chunks()
        logger.info(f"Loaded cached index ({len(chunks)} chunks)")
        return vectorstore, chunks

    if cached is not None:
        logger.info("Sources changed since the last build, rebuilding the index")

    all_documents = load_pdfs()
    load_links(all_documents)
    chunks = chunk_documents(all_documents)
    logger.info(f"Created {len(chunks)} chunks")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    save_chunks(chunks)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint, "chunks": len(chunks)}, f)
    logger.info("Built and saved new index")
    return vectorstore, chunks

def build_retriever(vectorstore, chunks, mode=DEFAULT_MODE, k=3, reranker=None):
    if mode == "hybrid_rerank" and reranker is None:
        reranker = CrossEncoderReranker()
    return HybridRetriever(vectorstore, chunks, mode=mode, k=k, reranker=reranker)

def build_llm(GROQ_API_KEY):

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
    return llm ,prompt


def build_pipeline(mode=DEFAULT_MODE):

    load_dotenv()

    # Get API key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in .env file. Please add it.")

    logger.info("GROQ API Key loaded successfully!")

    vectorstore, chunks = load_or_build_index()
    llm, prompt = build_llm(GROQ_API_KEY)

    retriever = build_retriever(vectorstore, chunks, mode=mode, k=3)
    logger.info(f"Retrieval mode: {mode}")
    return retriever, llm, prompt

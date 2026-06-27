# RAG Pipeline with LangChain & Groq

An end-to-end **Retrieval-Augmented Generation (RAG)** system built with LangChain, FAISS, and Groq's Llama 3.3 70B model.

## Features

- PDF document loading and intelligent chunking
- Vector embeddings using Hugging Face (all-MiniLM-L6-v2)
- FAISS vector store for fast similarity search
- Retrieval-Augmented Generation using Groq's powerful LLM
- Clean environment variable management

## Tech Stack

- **LangChain** – Orchestration framework
- **FAISS** – Vector database
- **Groq** – Fast inference with Llama 3.3 70B
- **Hugging Face Embeddings**
- **Python 3.10+**

## Project Structure
rag-pipeline/
├── data/                  # PDF documents
├── index/                 # FAISS vector index
├── .env                   # API keys (not committed)
├── main.py                # Main RAG pipeline
├── .gitignore
└── README.md
text## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/erraji-jihane/rag-pipeline.git
   cd rag-pipeline

Create virtual environment and install dependencies:Bashpython -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Add your Groq API key:envGROQ_API_KEY=gsk_...
Run the pipeline:Bashpython main.py

Future Improvements

Web UI with Gradio/Streamlit
Advanced chunking strategies
Evaluation metrics for retrieval quality
Support for multiple documents
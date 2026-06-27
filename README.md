# RAG Pipeline with LangChain & Groq

An **end-to-end Retrieval-Augmented Generation (RAG)** system built to query PDF documents using AI.

## Features

- PDF document loading and intelligent chunking
- Vector embeddings using Hugging Face
- FAISS vector store for fast similarity search
- High-quality responses powered by Groq (Llama 3.3 70B)
- Clean `.env` configuration

## Tech Stack

- **LangChain** – Framework
- **FAISS** – Vector database
- **Groq** – Fast LLM inference
- **Hugging Face Embeddings**
- **Python**

## Project Structure

```bash
rag-pipeline/
├── data/                    # PDF documents go here
├── index/                   # FAISS vector index (auto-generated)
├── main.py                  # Main RAG pipeline
├── .env                     # API keys (gitignored)
├── .gitignore
└── README.md
🚀 How to Run

Clone the repo:Bashgit clone https://github.com/erraji-jihane/rag-pipeline.git
cd rag-pipeline
Install dependencies:Bashpython -m venv .venv
.venv\Scripts\activate
pip install langchain langchain-community langchain-text-splitters langchain-huggingface langchain-groq faiss-cpu python-dotenv pypdf
Add your Groq key in .env:envGROQ_API_KEY=your_key_here
Run:Bashpython main.py

🔮 Future Improvements

Web UI using Gradio or Streamlit
Support for multiple documents
Advanced chunking strategies
Retrieval evaluation metrics
Conversation memory
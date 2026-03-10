# AI Generative - RAG Demos (LangChain + Chainlit)

This repository contains Retrieval-Augmented Generation (RAG) examples using OpenAI embeddings, FAISS vector search, and LangChain.

## Project Contents

- `rag.py`: Minimal RAG script using OpenAI client + FAISS.
- `rag_langchain.py`: LangChain-based RAG with conversation memory.
- `chainlit_app.py`: Chainlit web app using `rag_chain` from `rag_langchain.py`.
- `RAG.ipynb`, `langchain.ipynb`, `langgraph.ipynb`, `agents.ipynb`: Notebook experiments.
- `Data/`: Place your source documents (`.txt`, `.pdf`) here or update `DOCS_DIR` in scripts.

## Requirements

Python 3.10+ is recommended.

Dependencies are listed in `requirements.txt`:

- chainlit
- faiss-cpu
- langchain
- langchain-classic
- langchain-community
- langchain-openai
- langchain-text-splitters
- numpy
- openai>=2.26.0,<3.0.0
- pypdf
- python-dotenv
- tqdm

## Setup

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the repo root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Configure Document Path

Before running, check `DOCS_DIR` in:

- `rag.py`
- `rag_langchain.py`

Set it to the folder containing your `.txt` and `.pdf` files.

Example:

```python
DOCS_DIR = Path(r"C:\Users\YourName\Desktop\ai_generative\Data")
```

## Run the LangChain CLI Script

```powershell
python .\rag_langchain.py
```

This will:

- load and split docs,
- create/load a FAISS store,
- start an interactive terminal chat loop.

## Run the Chainlit App

```powershell
chainlit run chainlit_app.py -w
```

Then open the local URL printed in the terminal (usually `http://localhost:8000`).

## Notes

- FAISS artifacts are stored in `faiss_store/` (LangChain flow) and `faiss.index` + `faiss.index.meta.json` (minimal script flow).
- If document loading fails, verify `DOCS_DIR` and confirm files exist with `.txt` or `.pdf` extension.
- If OpenAI errors occur, verify `.env` and API key permissions.

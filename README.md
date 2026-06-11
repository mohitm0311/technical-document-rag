<div align="center">

# 🧠 DocMind

### Technical Document RAG System

*Semantic question answering over technical documents — powered by vector search and local LLMs*

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0070F3?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## Overview

**DocMind** is a Retrieval-Augmented Generation (RAG) system that enables intelligent, grounded question answering over technical documents. Upload any PDF, automatically build a semantic vector index, and query it in natural language — with source citations and similarity scores for every answer.

Built as a hands-on exploration of modern RAG pipelines, vector search, sentence embeddings, and local LLM inference.

---

## Features

- **PDF Ingestion** — Upload and process any PDF document
- **Sentence-Aware Chunking** — Smart text splitting that preserves semantic boundaries
- **Semantic Embeddings** — Dense vector representations via SentenceTransformers
- **FAISS Vector Index** — Millisecond-speed similarity search at scale
- **Local LLM Inference** — Privacy-preserving generation using Ollama + Qwen
- **Source Attribution** — Every answer cites the retrieved chunks with similarity scores
- **Interactive Dashboard** — Clean Streamlit UI with real-time indexing
- **Query History** — Track and revisit previous questions

---

## Architecture

```
PDF Document
     │
     ▼
PDF Loader (PyMuPDF)
     │
     ▼
Sentence-Aware Chunker
     │
     ▼
SentenceTransformer Embeddings
     │
     ▼
FAISS Vector Index
     │
     ▼
Top-K Retriever
     │
     ▼
Prompt Builder
     │
     ▼
Qwen LLM via Ollama
     │
     ▼
Answer + Source Citations
```

---

## Tech Stack

| Component        | Technology              |
|------------------|-------------------------|
| Language         | Python 3.10+            |
| UI               | Streamlit               |
| PDF Processing   | PyMuPDF                 |
| Embeddings       | SentenceTransformers    |
| Vector Search    | FAISS                   |
| LLM              | Qwen 3 (via Ollama)     |
| Retrieval        | Custom Retriever        |
| Chunking         | Sentence-Based Chunking |

---

## Project Structure

```
technical-document-rag/
│
├── data/
│   ├── pdfs/                  # Sample/test documents
│   └── uploads/               # User-uploaded PDFs
│
├── src/
│   ├── ingestion/             # PDF loading and text extraction
│   ├── chunking/              # Sentence-aware text splitting
│   ├── embeddings/            # SentenceTransformer wrapper
│   ├── retrieval/             # FAISS index and retriever
│   ├── rag/                   # RAG pipeline orchestration
│   ├── llm/                   # Ollama LLM interface
│   └── utils/                 # Shared helpers
│
├── tests/                     # Unit tests
├── app.py                     # Streamlit entry point
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 1. Clone the Repository

```bash
git clone https://github.com/mohitm0311/technical-document-rag.git
cd technical-document-rag
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
```

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up the LLM

Install Ollama from [ollama.com](https://ollama.com), then pull the Qwen model:

```bash
ollama pull qwen3:4b
ollama serve
```

### 5. Launch the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How It Works

1. **Upload** a PDF document via the Streamlit dashboard
2. **Extract** text using PyMuPDF
3. **Chunk** text into sentence-aware segments
4. **Embed** each chunk using SentenceTransformers
5. **Index** all embeddings in a FAISS vector store
6. **Query** — enter a natural language question
7. **Retrieve** the top-k most semantically similar chunks
8. **Generate** a grounded answer using the Qwen LLM
9. **Cite** — source chunks and similarity scores are displayed alongside every answer

---

## Example Queries

```
What is K-Means Clustering?
Explain how Gradient Boosting works.
What are the assumptions of KNN?
Summarize the chapter on Neural Networks.
Compare Random Forest and Gradient Boosting.
```

---

## Screenshots

| Dashboard | Retrieval Results |
|-----------|-------------------|
| *Add screenshot* | *Add screenshot* |

---

## Roadmap

- [ ] Page-level source citations
- [ ] Multi-document retrieval
- [ ] Persistent FAISS index storage
- [ ] Hybrid search (BM25 + FAISS)
- [ ] Conversation memory
- [ ] Reranking with cross-encoder models
- [ ] Cloud deployment

---

## Author

**Mohit Mehto**
B.Tech Chemical Engineering — Indian Institute of Technology Delhi

---

<div align="center">
  <sub>Built to understand RAG systems, vector search, embeddings, and local LLM deployment from the ground up.</sub>
</div>

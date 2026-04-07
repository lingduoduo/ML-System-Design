# RAG Sample System

This repository is a compact Retrieval-Augmented Generation (RAG) example in Python. It now supports two operating modes:

- a lightweight in-memory teaching pipeline that works with minimal dependencies
- an optional LangChain + FAISS + local Hugging Face path for more realistic document retrieval

The code is organized so the FastAPI app can boot with the simple fallback path, but automatically use the richer retrieval stack when the required packages and local data are available.

## What The System Does

The application walks through the core stages of a RAG system:

- collect or load source documents
- split them into chunks
- build an embedding-backed retrieval index
- retrieve relevant context for a user query
- construct a prompt and generate a response
- log requests for monitoring

The main composition happens in [rag_system.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/rag_system.py).

## Architecture

### Core modules

1. [data_collection_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/data_collection_pipeline.py) builds a small in-memory document store from sample source data.
2. [feature_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/feature_pipeline.py) contains:
   - the toy feature pipeline
   - retriever implementations for `dense`, `bm25`, and `hnsw`
   - query expansion
   - optional LangChain/FAISS builders
   - local Hugging Face embedding and LLM helpers
3. [retriever.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/retriever.py) exposes retrieval clients for both the toy index and the LangChain retriever.
4. [Inference_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/Inference_pipeline.py) builds the final prompt and invokes the model.
5. [monitoring.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/monitoring.py) stores request history in memory.
6. [deploy.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/deploy.py) returns simple deployment metadata.
7. [serving.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/serving.py) serves the API and caches the initialized system.

### Startup behavior

When [rag_system.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/rag_system.py) builds the app state, it:

1. creates the sample in-memory document store
2. builds the toy feature store and fallback model path
3. tries to load markdown files from `./data/` with LangChain
4. tries to split those files, embed them, build a FAISS store, and load a local Hugging Face LLM
5. uses the LangChain path if that succeeds
6. falls back to the in-memory toy path if dependencies or files are missing

That means the API remains usable even in a minimal environment.

## Retrieval Modes

The toy retrieval stack supports three retrieval strategies:

- `dense`: embedding-based dense retrieval over the in-memory vector matrix
- `bm25`: lexical retrieval using BM25-style scoring
- `hnsw`: an approximate dense retrieval path with a lightweight HNSW-style neighbor graph

There is also optional query expansion, which generates a few synonym-based query variants before retrieval and merges the results.

## Repository Layout

```text
RAG/
├── README.md
├── serving.py
├── rag_system.py
├── data_collection_pipeline.py
├── feature_pipeline.py
├── retriever.py
├── training_pipeline.py
├── Inference_pipeline.py
├── deploy.py
└── monitoring.py
```

## Requirements

### Minimal setup

This is enough to run the toy in-memory pipeline:

```bash
pip install fastapi uvicorn pydantic numpy
```

### Optional LangChain + FAISS + local HF setup

Install these if you want the real document-loading and local-model path:

```bash
pip install fastapi uvicorn pydantic numpy \
  langchain-community langchain-text-splitters \
  faiss-cpu sentence-transformers transformers torch
```

Python 3.10+ is recommended.

## Local Data

The optional LangChain pipeline expects markdown files under `./data/`:

```text
data/
├── doc1.md
├── doc2.md
└── ...
```

If `./data/` is missing, empty, or the LangChain dependencies are unavailable, the application will fall back to the toy in-memory dataset.

## Run The API

From the repository root:

```bash
uvicorn serving:app --reload
```

Endpoints:

- `GET /health`
- `POST /generate`

## API

### `GET /health`

Returns basic service metadata:

- status
- number of source documents
- number of indexed chunks
- deployment metadata
- available retrieval modes

### `POST /generate`

Request body:

```json
{
  "query": "What is a RAG pipeline?",
  "top_k": 2,
  "retrieval_mode": "dense",
  "expand_query": false
}
```

Request fields:

- `query`: user query
- `top_k`: number of chunks to retrieve
- `retrieval_mode`: one of `dense`, `bm25`, or `hnsw`
- `expand_query`: whether to run synonym-based query expansion first

Example:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query":"What is a rag pipeline?",
    "top_k": 3,
    "retrieval_mode":"bm25",
    "expand_query": true
  }'
```

Example response shape:

```json
{
  "query": "What is a rag pipeline?",
  "retrieval_mode": "bm25",
  "expand_query": true,
  "retrieved_context": [
    "How retrieval augmented generation works.",
    "Example RAG pipeline with embeddings and vector DB."
  ],
  "response": "..."
}
```

## LangChain / HF Flow

The optional richer path in [feature_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/feature_pipeline.py) performs this sequence:

1. load markdown documents with `DirectoryLoader`
2. split them with `RecursiveCharacterTextSplitter`
3. build embeddings with `HuggingFaceEmbeddings`
4. create a FAISS vector store
5. expose a retriever with `k = TOP_K`
6. load a local Hugging Face text-generation model

This lets the project move beyond the toy embedder and toy vector store when you want a more realistic local setup.

## Current Limitations

Some parts are intentionally simplified:

- the fallback document store is in memory only
- the fallback model is still a placeholder fine-tuned model object
- the `hnsw` retriever is an educational approximation, not a production ANN library
- monitoring is stored only in process memory
- deployment metadata is simulated
- the LangChain path currently uses a basic adapter around the retriever and local LLM

## Recent Improvements

The codebase has been refactored to be easier to extend:

- removed import-time demo execution from modules
- centralized app assembly in [rag_system.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/rag_system.py)
- cached the initialized system in [serving.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/serving.py)
- added three retrieval modes: `dense`, `bm25`, and `hnsw`
- added query expansion
- added optional LangChain, FAISS, and local Hugging Face integration
- kept a graceful fallback path for minimal environments

## Quick Start For Developers

Inspect the assembled system in Python:

```python
from rag_system import build_rag_system

system = build_rag_system()
print(type(system.retrieval_client).__name__)
print(system.deployment_info)
```

If the optional dependencies are installed and `./data/` is populated, the system will prefer the LangChain-backed path. Otherwise it will use the toy retrieval pipeline.

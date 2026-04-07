# RAG Sample System

This repository is a compact Retrieval-Augmented Generation (RAG) example written in Python. It walks through the main stages of a simple RAG stack:

- collecting source documents
- transforming them into chunks and embeddings
- building a retrieval index
- training and registering a toy language model
- serving responses through a FastAPI app
- logging request activity

The current implementation uses lightweight in-memory components so the architecture is easy to follow and modify.

## Project Flow

The system is composed in [`rag_system.py`](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/rag_system.py).

1. [`data_collection_pipeline.py`](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/data_collection_pipeline.py) builds an in-memory document store from sample Medium-style articles and GitHub README-style content.
2. [`feature_pipeline.py`](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/feature_pipeline.py) chunks document text, generates toy embeddings, and builds a simple vector database plus an instruction dataset.
3. [`training_pipeline.py`](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/training_pipeline.py) fine-tunes a placeholder model, evaluates it, and registers the accepted model.
4. [`retriever.py`](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/retriever.py) converts user queries into embeddings and searches the vector store.
5. [`Inference_pipeline.py`](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/Inference_pipeline.py) builds a prompt from retrieved context and asks the model to generate an answer.
6. [`deploy.py`](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/deploy.py) exposes simple deployment metadata.
7. [`monitoring.py`](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/monitoring.py) logs request history in memory.
8. [`serving.py`](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/RAG/serving.py) exposes the FastAPI endpoints and caches the initialized RAG system.

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

Install these Python packages in your environment:

```bash
pip install fastapi uvicorn pydantic numpy
```

Python 3.10+ is recommended.

## Run The API

From the repository root:

```bash
uvicorn serving:app --reload
```

Once the server is running:

- health check: `GET /health`
- generation endpoint: `POST /generate`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query":"What is a RAG pipeline?","top_k":2}'
```

Example response shape:

```json
{
  "query": "What is a RAG pipeline?",
  "retrieved_context": [
    "How retrieval augmented generation works.",
    "Example RAG pipeline with embeddings and vector DB."
  ],
  "response": "[base-llm-finetuned] Response to: ..."
}
```

## API Endpoints

### `GET /health`

Returns a summary of the current in-memory system state:

- service status
- number of source documents
- number of indexed chunks
- deployment metadata

### `POST /generate`

Request body:

```json
{
  "query": "What is a RAG pipeline?",
  "top_k": 2
}
```

Behavior:

- retrieves the top matching chunks
- constructs a prompt from the retrieved context
- generates a response with the accepted model
- logs the interaction in the monitor

## Design Notes

This codebase is intentionally small and educational. A few implementation details are simplified:

- the document store is an in-memory dictionary
- embeddings are deterministic random vectors
- vector search uses NumPy dot products over an in-memory matrix
- model fine-tuning and evaluation are placeholders
- monitoring is stored only in process memory
- deployment metadata is simulated

Even with those simplifications, the separation between collection, features, retrieval, training, inference, deployment, and serving mirrors how larger production systems are often organized.

## Recent Improvements

The current version has already been cleaned up to make the system easier to extend:

- removed import-time execution from pipeline modules
- centralized system construction in `rag_system.py`
- cached the initialized RAG stack in the FastAPI layer
- added vectorized search in the toy vector store
- added embedding caching to avoid repeated computation
- made the modules reusable as building blocks instead of demo scripts

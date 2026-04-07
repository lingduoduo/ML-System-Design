# RAG Sample System

This repository is a compact Retrieval-Augmented Generation (RAG) example in Python. It supports two operating modes:

- a lightweight in-memory teaching pipeline with minimal dependencies
- an optional LangChain + FAISS + local Hugging Face pipeline for more realistic retrieval and generation

The FastAPI app will try to use the richer local stack when dependencies and local data are available, and otherwise fall back to the toy in-memory path.

## What The System Does

The project walks through the main stages of a RAG system:

- load or assemble documents
- split them into chunks
- build retrieval indices
- retrieve relevant context
- optionally rewrite the question
- optionally run a multi-step inference agent
- generate an answer
- monitor and evaluate behavior

The main assembly happens in [rag_system.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/rag_system.py).

## Architecture

### Core modules

1. [data_collection_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/data_collection_pipeline.py) builds a small in-memory document store from sample source data.
2. [feature_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/feature_pipeline.py) contains:
   - the toy feature pipeline
   - retriever implementations for `dense`, `bm25`, and `hnsw`
   - query expansion
   - optional LangChain/FAISS feature-store builders
   - local Hugging Face embedding and LLM helpers
   - question rewriting helpers
3. [retriever.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/retriever.py) exposes retrieval clients for both the toy index and the LangChain retriever, plus a hybrid `MultiPathRetriever` that fuses lexical and vector search.
4. [Inference_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/Inference_pipeline.py) contains:
   - prompt construction
   - `LLMTwin`
   - LangChain RAG chain construction
   - `RAGQueryEngine` for retrieve -> rerank -> answer
   - `CrossEncoderReranker` for optional query-document reranking
   - `QueryUnderstandingEngine` for rewrite, HyDE, and decomposition
   - `RetrievalToolset` for domain retrieval actions
   - multi-step agent planning and execution
5. [evaluation.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/evaluation.py) contains:
   - evaluation tracking
   - evaluation dataset utilities
   - judge-based evaluation helpers
   - retrieval evaluation metrics
6. [deploy.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/deploy.py) returns simple deployment metadata.
7. [serving.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/serving.py) exposes the FastAPI API and caches the initialized system.

The richer LangChain-backed runtime is assembled through `RichRAGBuilder` in [rag_system.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/rag_system.py), which keeps loading, hybrid retrieval, reranking, and agent setup in one place.

### Startup behavior

When [rag_system.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/rag_system.py) builds the app state, it:

1. creates the sample in-memory document store
2. builds the toy feature store and fallback model path
3. tries to load markdown files from `./data/`
4. tries to build a LangChain feature store, hybrid retrievers, a rewrite chain, a RAG chain, and a multi-step inference agent
5. uses that richer stack if initialization succeeds
6. falls back to the toy pipeline if dependencies or data are missing

That keeps the app usable in a minimal environment while still supporting a richer local stack.

## Retrieval Modes

The toy retrieval stack supports three strategies:

- `dense`: embedding-based dense retrieval over the in-memory vector matrix
- `bm25`: lexical retrieval using BM25-style scoring
- `hnsw`: an approximate dense retrieval path with a lightweight HNSW-style neighbor graph

Query expansion is also available and generates a few synonym-based variants before retrieval. In the richer LangChain path, retrieval can also combine multiple data paths, such as dense vector search and BM25 lexical search.

In the current LangChain-backed setup, hybrid retrieval is implemented with `MultiPathRetriever`, which:

- runs BM25 and vector retrieval in parallel paths
- fuses the scores with configurable weights
- deduplicates overlapping results
- respects runtime `top_k` limits from the API layer
- keeps the strongest items and annotates metadata with retrieval provenance

For standard answer generation, the richer runtime can also route queries through `RAGQueryEngine`, which:

- retrieves documents from the hybrid retriever
- optionally reranks them with the `BAAI/bge-reranker-base` cross-encoder
- builds a bounded context window
- generates the final answer from that context

## Inference Features

The inference layer supports three levels of behavior:

- standard single-step answer generation
- optional question rewriting before retrieval
- optional multi-step planning with ambiguity-aware query rewriting

Multi-step querying now lives in [Inference_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/Inference_pipeline.py), not in a separate module.

In the current implementation, the multi-step agent can:

- use three query-understanding strategies: query rewrite, multi-question decomposition, and HyDE-style hypothetical expansion
- detect when the current request is ambiguous or too conversational for tool calls
- rewrite it into a shorter, tool-friendly working query without adding new details
- decompose bundled requests into smaller retrieval subquestions
- run domain-specific retrieval tools
- combine tool outputs into a final answer

### Optional LangChain + FAISS + local HF setup

Install these if you want the richer local retrieval and inference stack:

```bash
pip install fastapi uvicorn pydantic numpy \
  langchain-community langchain-core langchain-text-splitters \
  langgraph faiss-cpu sentence-transformers transformers torch
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

If `./data/` is missing, empty, or the optional dependencies are unavailable, the application will fall back to the toy dataset and toy retriever path.

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
- whether multi-step querying is available in the current runtime

### `POST /generate`

Request body:

```json
{
  "query": "What is a RAG pipeline?",
  "top_k": 2,
  "retrieval_mode": "dense",
  "expand_query": false,
  "rewrite_question": false,
  "multi_step": false
}
```

Request fields:

- `query`: user query
- `top_k`: number of chunks to retrieve
- `retrieval_mode`: one of `dense`, `bm25`, or `hnsw`
- `expand_query`: whether to run synonym-based query expansion
- `rewrite_question`: whether to rewrite the query before retrieval
- `multi_step`: whether to run the multi-step inference agent instead of the standard retrieval flow

Example standard retrieval request:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query":"What is a rag pipeline?",
    "top_k": 3,
    "retrieval_mode":"bm25",
    "expand_query": true,
    "rewrite_question": true,
    "multi_step": false
  }'
```

Example multi-step request:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query":"I want somewhere good for kids and maybe food nearby.",
    "multi_step": true
  }'
```

Example response shape:

```json
{
  "query": "What is a rag pipeline?",
  "rewritten_query": "What is retrieval augmented generation pipeline?",
  "retrieval_mode": "bm25",
  "expand_query": true,
  "rewrite_question": true,
  "multi_step": false,
  "retrieved_context": [
    "How retrieval augmented generation works.",
    "Example RAG pipeline with embeddings and vector DB."
  ],
  "response": "..."
}
```

## LangChain / HF Flow

The optional richer path uses the following sequence:

1. load markdown documents with `DirectoryLoader`
2. split them with `RecursiveCharacterTextSplitter`
3. build embeddings with `HuggingFaceEmbeddings`
4. create a FAISS vector store
5. build a BM25 retriever over the same chunk set
6. load a local Hugging Face text-generation model
7. fuse vector and lexical results through a multi-path retriever
8. rerank the fused candidates with the local LLM
9. build a question rewrite chain for retrieval
10. build a RAG chain
11. build a query engine for retrieve -> rerank -> answer
12. build a multi-step inference agent with query rewrite, multi-question decomposition, HyDE-style disambiguation, and domain tools

This lets the project move beyond the toy embedder and toy vector store while also supporting richer inference behavior.

## Evaluation

[evaluation.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/evaluation.py) now includes:

- runtime request logging with `EvaluationTracker` and a backward-compatible `Monitor` alias
- evaluation examples and datasets
- Recall@K utilities
- judge-chain construction
- answer judging
- end-to-end dataset evaluation with `RAGEvaluator`
- optional RAGAS-based evaluation with `RagasEvaluator`
- retrieval-method comparison utilities with `RetrievalMethodComparison`
- chunking-strategy experiments with `ChunkingStrategyEvaluator`

## Current Limitations

Some parts are intentionally simplified:

- the fallback document store is in memory only
- the fallback model is still a placeholder fine-tuned model object
- the `hnsw` retriever is an educational approximation, not a production ANN library
- evaluation history is stored only in process memory
- deployment metadata is simulated
- multi-step querying depends on optional LangChain and LangGraph support
- ambiguity handling is prompt-based, so rewrite quality depends on the local LLM
- the LangChain path still uses lightweight adapters around retrievers and models
- RAGAS evaluation is optional and requires extra packages plus an appropriate evaluation LLM setup
- retrieval-method comparison utilities also require `sentence-transformers`, `scikit-learn`, `numpy`, and `torch`
- chunking-strategy experiments also require `langchain-openai` and suitable API credentials

## Recent Improvements

The codebase has been refactored to be easier to extend:

- removed import-time demo execution from modules
- centralized app assembly in [rag_system.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/rag_system.py)
- cached the initialized system in [serving.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/serving.py)
- added three retrieval modes: `dense`, `bm25`, and `hnsw`
- added hybrid multi-path retrieval for the LangChain-backed path
- added query-engine reranking with an optional cross-encoder stage
- improved LangChain retrieval so API `top_k` is honored consistently
- removed redundant embedding initialization during rich-stack startup
- added query expansion
- added question rewriting
- moved multi-step agent behavior into [Inference_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/Inference_pipeline.py)
- added ambiguity-first query rewriting for tool-friendly multi-step planning
- clarified the three query-understanding strategies: rewrite, multi-question decomposition, and HyDE
- refactored the inference and runtime setup around dedicated classes for query understanding, tool execution, and rich-runtime assembly
- added optional LangChain, FAISS, local Hugging Face, and LangGraph integration
- renamed the evaluation module to [evaluation.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/evaluation.py) and kept [monitoring.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/monitoring.py) as a compatibility shim
- added an optional RAGAS-style evaluation path for answer/context quality metrics
- kept a graceful fallback path for minimal environments

## Quick Start For Developers

Inspect the assembled system in Python:

```python
from rag_system import build_rag_system

system = build_rag_system()
print(type(system.retrieval_client).__name__)
print(system.rewrite_chain is not None)
print(system.multi_step_agent is not None)
print(system.deployment_info)
```

If the optional dependencies are installed and `./data/` is populated, the system will prefer the LangChain-backed path. Otherwise it will use the toy retrieval pipeline.

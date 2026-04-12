# RAG Sample System

This repository is a compact, code-first reference implementation of a Retrieval-Augmented Generation (RAG) system in Python.

It is designed to show the moving parts of a RAG stack without requiring a full production environment:

- a toy in-memory pipeline that works with minimal dependencies
- an optional richer local pipeline built on LangChain, FAISS, BM25, and local Hugging Face models
- a FastAPI service that exposes the system through simple HTTP endpoints
- evaluation, training, routing, and deployment stubs that make the repo feel like a small end-to-end ML system instead of a single script

The app automatically tries to boot the richer stack first and falls back to the toy path if optional dependencies or local data are missing.

## Why This Repo Exists

This repo is useful if you want to study or demo:

- how documents move from collection to chunking to retrieval
- how different retrieval modes can sit behind one API
- how query rewriting and multi-step reasoning can be layered onto RAG
- how to organize a small RAG codebase into pipelines instead of one notebook or script
- how to keep a service usable even when the "full" stack is unavailable

This is not a production-ready RAG platform. It is a teaching and prototyping repo with clear seams for extension.

## How The System Runs

There are two runtime modes.

### 1. Fallback teaching mode

This path always works as long as the basic Python dependencies are available.

It uses:

- a tiny in-memory document store from [data_collection_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/data_collection_pipeline.py)
- a lightweight feature pipeline and vector store from [feature_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/feature_pipeline.py)
- a simple retrieval client from [retriever.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/retriever.py)
- a placeholder fine-tuned model path from [training_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/training_pipeline.py)

This mode is good for understanding the control flow of the system.

### 2. Rich local mode

This path activates when:

- optional LangChain / FAISS / transformer dependencies are installed
- markdown files are available in `./data/`

It adds:

- document loading and chunking from local markdown files
- Hugging Face embeddings
- a FAISS vector store
- BM25 + vector hybrid retrieval
- optional query rewriting
- optional cross-encoder reranking
- a standard query engine for retrieve -> rerank -> answer
- an optional multi-step agent path

The runtime assembly happens in [rag_system.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/rag_system.py).

## Workflow

You can describe the system with the following high-level workflow:

```text
User question
   ↓
[Retrieval] -> FAISS + Sentence Transformers (lightweight local retrieval, with fallback to the toy retriever)
   ↓
[Generation] -> GPT / fine-tuned model (in this repo, mainly a local HF model or placeholder fine-tuned model)
   ↓
[Validation] -> Prompt-based grounded / consistency checks (light hallucination resistance, not a standalone fact verifier)
   ↖______↓______↗
         |
   [LangGraph] <- Flow control: continue tool use, rewrite the query, decompose subquestions, retry
         ↓
   [Evaluation] <- EvaluationTracker + judge / RAGAS utilities (LangSmith can be added later)
         ↓
   [Serving] <- FastAPI (standard API interface)
```

How that maps to this repo:

- `[Retrieval]`: the rich path uses FAISS and `sentence-transformers`; the fallback path uses the in-memory retriever in [feature_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/feature_pipeline.py) and [retriever.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/retriever.py)
- `[Generation]`: answers come from the `LLMTwin`, the query engine, or the multi-step agent in [Inference_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/Inference_pipeline.py)
- `[Validation]`: the repo now includes a lightweight validation pass that checks lexical grounding against retrieved context and flags low-support answers; it is still heuristic, not a standalone factual verification service
- `[LangGraph]`: the multi-step planner/tool loop is implemented in [Inference_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/Inference_pipeline.py) when optional dependencies are installed
- `[Evaluation]`: lightweight tracing/evaluation exists in [evaluation.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/evaluation.py)
- `[LangSmith]`: not currently integrated; if you want, it should be described as a future observability/evaluation extension rather than an existing feature
- `[Serving]`: FastAPI endpoints live in [serving.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/serving.py)

## Request Flow

At a high level, a request moves through the system like this:

1. [serving.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/serving.py) receives the HTTP request.
2. [llm_gateway.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/llm_gateway.py) chooses an execution path: `fast`, `balanced`, or `premium`.
3. The system may rewrite the query if rewriting is enabled and the rich runtime is available.
4. Retrieval runs through either the toy retriever or the LangChain-backed hybrid retriever.
5. The selected answering path generates a response:
   - direct retrieval + answer generation
   - query-engine-based RAG
   - multi-step agent execution
6. A lightweight validation pass scores how well the answer is supported by retrieved context.
7. [evaluation.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/evaluation.py) logs the run in memory for lightweight monitoring.

## Repo Map

The codebase is split by responsibility rather than framework layers.

- [serving.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/serving.py): FastAPI app, request/response models, cached system startup, `/health`, `/stats`, and `/generate`
- [rag_system.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/rag_system.py): top-level system assembly and fallback logic
- [llm_gateway.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/llm_gateway.py): routes requests to `fast`, `balanced`, or `premium` execution paths
- [data_collection_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/data_collection_pipeline.py): sample source ingestion, cleaning, ETL, and in-memory document store creation
- [feature_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/feature_pipeline.py): chunking, embeddings, toy index construction, LangChain feature-store setup, and query rewriting helpers
- [retriever.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/retriever.py): toy retrieval client, LangChain retrieval adapter, and hybrid `MultiPathRetriever`
- [Inference_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/Inference_pipeline.py): prompt construction, memory helpers, context templates, RAG query engine, reranking, and multi-step agent behavior
- [training_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/training_pipeline.py): placeholder model training, evaluation, registry, and experiment tracking
- [evaluation.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/evaluation.py): runtime logging plus optional evaluation utilities and experiments
- [deploy.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/deploy.py): simple deployment metadata stub

## Retrieval Modes

The fallback retriever supports three retrieval modes behind one API:

- `dense`: dense similarity over the toy embedding/vector path
- `bm25`: lexical retrieval
- `hnsw`: an educational approximation of ANN-style retrieval

In the richer runtime, the system can also use a hybrid retriever that combines:

- BM25 lexical recall
- vector similarity search over FAISS
- score fusion and deduplication
- optional reranking

## Gateway Tiers

The API supports model selection through the gateway instead of exposing raw model internals.

- `fast`: low-latency direct retrieval + answer generation
- `balanced`: standard RAG query engine when available
- `premium`: prefers the multi-step agent path when available
- `auto`: lets the gateway choose based on query complexity

This behavior lives in [llm_gateway.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/llm_gateway.py).

## Quick Start

### Install minimal dependencies

```bash
pip install fastapi uvicorn pydantic numpy
```

### Run the API

```bash
uvicorn serving:app --reload
```

### Check health

```bash
curl http://127.0.0.1:8000/health
```

### Generate an answer

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is a RAG pipeline?",
    "top_k": 3,
    "retrieval_mode": "bm25",
    "expand_query": true,
    "rewrite_question": false,
    "multi_step": false
  }'
```

## Optional Rich Local Setup

Install these packages if you want the richer local retrieval and inference stack:

```bash
pip install fastapi uvicorn pydantic numpy \
  langchain-community langchain-core langchain-text-splitters \
  langgraph faiss-cpu sentence-transformers transformers torch
```

Python 3.10+ is recommended.

Add markdown files under `./data/`:

```text
data/
├── doc1.md
├── doc2.md
└── ...
```

If the optional packages are missing, or `./data/` is absent or empty, the app will continue running on the fallback path.

## API

### `GET /health`

Returns a small snapshot of system state, including:

- service status
- current runtime mode: `rich` or `fallback`
- number of source documents
- number of indexed chunks
- deployment metadata
- supported retrieval modes
- whether multi-step execution is available
- gateway engine types

### `GET /stats`

Returns gateway usage stats such as request counts, engine selection counts, and lightweight validation counts.

### `POST /generate`

Request body:

```json
{
  "query": "What is a RAG pipeline?",
  "model": "auto",
  "top_k": 2,
  "retrieval_mode": "dense",
  "expand_query": false,
  "rewrite_question": false,
  "multi_step": false
}
```

Request fields:

- `query`: the user question
- `model`: `auto`, `fast`, `balanced`, or `premium`
- `top_k`: number of retrieved chunks, validated to `1..20`
- `retrieval_mode`: `dense`, `bm25`, or `hnsw`
- `expand_query`: whether to use query expansion
- `rewrite_question`: whether to rewrite the query before retrieval when supported
- `multi_step`: whether to prefer the multi-step path

Response fields also include:

- `validation`: a lightweight grounding summary with `grounded`, `confidence`, overlap scores, warnings, and a short summary

Example multi-step request:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want somewhere good for kids and maybe food nearby.",
    "model": "premium",
    "multi_step": true
  }'
```

Example response shape:

```json
{
  "query": "What is a RAG pipeline?",
  "rewritten_query": null,
  "model": "auto",
  "selected_model": "rag-query-engine",
  "selected_engine": "balanced",
  "selected_tier": "Balanced",
  "retrieval_mode": "dense",
  "expand_query": false,
  "rewrite_question": false,
  "multi_step": false,
  "retrieved_context": [
    "How retrieval augmented generation works."
  ],
  "validation": {
    "grounded": true,
    "confidence": "medium",
    "answer_context_overlap": 0.32,
    "query_context_overlap": 0.5,
    "warnings": [],
    "summary": "The answer appears reasonably grounded in the retrieved context."
  },
  "response": "..."
}
```

## Evaluation And Training Utilities

This repo also includes supporting pieces that make it useful for demos and interviews:

- `EvaluationTracker` logs requests and response previews in memory
- evaluation helpers support recall-style retrieval checks and judge-based evaluation
- training utilities simulate fine-tuning, experiment tracking, and model registration

These pieces are intentionally lightweight, but they show where those concerns belong in a larger system.

## Limitations

Some important things are simplified on purpose:

- the fallback database is in memory only
- the fallback model path is a placeholder object, not a real trained model
- deployment is simulated
- evaluation history is stored in process memory only
- the `hnsw` path is educational, not production ANN infrastructure
- the richer runtime depends on optional local packages and local model availability
- query rewriting and multi-step behavior are prompt-driven and only as good as the underlying local model

## Good Entry Points

If you are reading the repo for the first time, start here:

1. [serving.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/serving.py) to see the API surface
2. [rag_system.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/rag_system.py) to understand startup and fallback behavior
3. [llm_gateway.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/llm_gateway.py) to see how requests are routed
4. [retriever.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/retriever.py) and [feature_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/feature_pipeline.py) for retrieval details
5. [Inference_pipeline.py](/Users/linghuang/Git/ML-System-Design/sampled-ml-code/rag/Inference_pipeline.py) for prompt and agent logic

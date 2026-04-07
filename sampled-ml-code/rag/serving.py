from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

from feature_pipeline import RetrievalMode
from rag_system import RAGSystem, build_rag_system


app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 2
    retrieval_mode: RetrievalMode = "dense"
    expand_query: bool = False


@lru_cache(maxsize=1)
def get_system() -> RAGSystem:
    return build_rag_system()


@app.get("/health")
def healthcheck() -> dict:
    system = get_system()
    return {
        "status": "ok",
        "documents": len(system.db.find_all()),
        "chunks": len(system.vector_db.rows),
        "deployment": system.deployment_info,
        "retrieval_modes": ["dense", "bm25", "hnsw"],
    }


@app.post("/generate")
def generate(req: QueryRequest) -> dict:
    system = get_system()
    retrieved = system.retrieval_client.retrieve(
        req.query,
        top_k=req.top_k,
        mode=req.retrieval_mode,
        expand_query=req.expand_query,
    )
    response = system.llm_twin.answer(req.query, retrieved)
    system.monitor.log_request(req.query, retrieved, response)

    return {
        "query": req.query,
        "retrieval_mode": req.retrieval_mode,
        "expand_query": req.expand_query,
        "retrieved_context": [metadata["text"] for _, metadata in retrieved],
        "response": response,
    }

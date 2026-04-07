from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

from feature_pipeline import RetrievalMode, rewrite_question
from Inference_pipeline import run_multi_step_search
from rag_system import RAGSystem, build_rag_system


app = FastAPI()


def _count_chunks(system: RAGSystem) -> int:
    if system.langchain_feature_store is not None:
        return len(system.langchain_feature_store.chunks)
    if hasattr(system.vector_db, "rows"):
        return len(system.vector_db.rows)
    return 0


class QueryRequest(BaseModel):
    query: str
    top_k: int = 2
    retrieval_mode: RetrievalMode = "dense"
    expand_query: bool = False
    rewrite_question: bool = False
    multi_step: bool = False


@lru_cache(maxsize=1)
def get_system() -> RAGSystem:
    return build_rag_system()


@app.get("/health")
def healthcheck() -> dict:
    system = get_system()
    return {
        "status": "ok",
        "documents": len(system.db.find_all()),
        "chunks": _count_chunks(system),
        "deployment": system.deployment_info,
        "retrieval_modes": ["dense", "bm25", "hnsw"],
        "multi_step_available": system.multi_step_agent is not None,
    }


@app.post("/generate")
def generate(req: QueryRequest) -> dict:
    system = get_system()
    if req.multi_step:
        if system.multi_step_agent is None:
            return {
                "query": req.query,
                "rewritten_query": None,
                "retrieval_mode": req.retrieval_mode,
                "expand_query": req.expand_query,
                "rewrite_question": req.rewrite_question,
                "multi_step": True,
                "retrieved_context": [],
                "response": "Multi-step querying is unavailable in the current runtime.",
            }

        response = run_multi_step_search(system.multi_step_agent, req.query)
        system.evaluator.record_run(req.query, [], response)
        return {
            "query": req.query,
            "rewritten_query": None,
            "retrieval_mode": req.retrieval_mode,
            "expand_query": req.expand_query,
            "rewrite_question": req.rewrite_question,
            "multi_step": True,
            "retrieved_context": [],
            "response": response,
        }

    retrieval_query = req.query
    if req.rewrite_question and system.rewrite_chain is not None:
        rewritten = rewrite_question(system.rewrite_chain, req.query)
        if not rewritten.startswith("An error occurred") and rewritten != "No valid rewritten result":
            retrieval_query = rewritten

    retrieved = system.retrieval_client.retrieve(
        retrieval_query,
        top_k=req.top_k,
        mode=req.retrieval_mode,
        expand_query=req.expand_query,
    )
    response = system.llm_twin.answer(retrieval_query, retrieved)
    system.evaluator.record_run(retrieval_query, retrieved, response)

    return {
        "query": req.query,
        "rewritten_query": retrieval_query if retrieval_query != req.query else None,
        "retrieval_mode": req.retrieval_mode,
        "expand_query": req.expand_query,
        "rewrite_question": req.rewrite_question,
        "multi_step": False,
        "retrieved_context": [metadata["text"] for _, metadata in retrieved],
        "response": response,
    }

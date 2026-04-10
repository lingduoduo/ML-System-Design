from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from feature_pipeline import RetrievalMode
from llm_gateway import LLMGateway
from rag_system import RAGSystem, build_rag_system


app = FastAPI()


def _count_chunks(system: RAGSystem) -> int:
    if system.langchain_feature_store is not None:
        return len(system.langchain_feature_store.chunks)
    if hasattr(system.vector_db, "rows"):
        return len(system.vector_db.rows)
    return 0


def _build_response_payload(
    *,
    req: "QueryRequest",
    query: str,
    response: str,
    retrieved_context: list[str],
    multi_step: bool,
    selected_model: str | None = None,
    selected_engine: str | None = None,
    selected_tier: str | None = None,
) -> dict:
    return {
        "query": req.query,
        "rewritten_query": query if query != req.query else None,
        "model": req.model,
        "selected_model": selected_model,
        "selected_engine": selected_engine,
        "selected_tier": selected_tier,
        "retrieval_mode": req.retrieval_mode,
        "expand_query": req.expand_query,
        "rewrite_question": req.rewrite_question,
        "multi_step": multi_step,
        "retrieved_context": retrieved_context,
        "response": response,
    }


class QueryRequest(BaseModel):
    query: str
    model: str = "auto"
    top_k: int = Field(default=2, ge=1, le=20)
    retrieval_mode: RetrievalMode = "dense"
    expand_query: bool = False
    rewrite_question: bool = False
    multi_step: bool = False


@lru_cache(maxsize=1)
def get_system() -> RAGSystem:
    return build_rag_system()


@lru_cache(maxsize=1)
def get_gateway() -> LLMGateway:
    return LLMGateway(get_system())


@app.get("/health")
def healthcheck() -> dict:
    system = get_system()
    gateway = get_gateway()
    return {
        "status": "ok",
        "documents": len(system.db.find_all()),
        "chunks": _count_chunks(system),
        "deployment": system.deployment_info,
        "retrieval_modes": ["dense", "bm25", "hnsw"],
        "multi_step_available": system.multi_step_agent is not None,
        "gateway_engines": gateway.stats()["engine_types"],
    }


@app.get("/stats")
def stats() -> dict:
    return get_gateway().stats()


@app.post("/generate")
def generate(req: QueryRequest) -> dict:
    system = get_system()
    gateway = get_gateway()

    requested_model = "premium" if req.multi_step and req.model == "auto" else req.model
    try:
        result = gateway.execute(
            query=req.query,
            top_k=req.top_k,
            retrieval_mode=req.retrieval_mode,
            expand_query=req.expand_query,
            rewrite_enabled=req.rewrite_question,
            requested_model=requested_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    system.evaluator.record_run(result.query, result.retrieved, result.response)

    return _build_response_payload(
        req=req,
        query=result.query,
        response=result.response,
        retrieved_context=result.retrieved_context,
        multi_step=result.used_multi_step,
        selected_model=result.selected_model,
        selected_engine=result.engine.name,
        selected_tier=result.engine.tier,
    )

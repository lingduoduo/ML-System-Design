from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from feature_pipeline import rewrite_question
from retriever import documents_to_retrieval_results


COMPLEX_WORDS = (
    "design",
    "architecture",
    "analysis",
    "system",
    "algorithm",
    "optimization",
)
MEDIUM_WORDS = ("explain", "principle", "method", "process")
SIMPLE_WORDS = ("what is", "definition", "translate")


@dataclass(frozen=True)
class GatewayEngine:
    name: str
    tier: str
    description: str


@dataclass
class GatewayExecutionResult:
    engine: GatewayEngine
    query: str
    response: str
    retrieved_context: List[str]
    retrieved: List[tuple[float, Dict[str, Any]]] = field(default_factory=list)
    used_rewrite: bool = False
    used_multi_step: bool = False


@dataclass
class LLMGateway:
    system: Any
    request_count: int = 0
    engine_counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for engine_name in ("fast", "balanced", "premium"):
            self.engine_counts.setdefault(engine_name, 0)

    def available_engines(self) -> Dict[str, GatewayEngine]:
        return {
            "fast": GatewayEngine(
                name="fast",
                tier="Fast",
                description="Low-latency direct retrieval + response generation.",
            ),
            "balanced": GatewayEngine(
                name="balanced",
                tier="Balanced",
                description="Standard RAG query engine with reranking when available.",
            ),
            "premium": GatewayEngine(
                name="premium",
                tier="Premium",
                description="Most capable path, preferring multi-step reasoning when available.",
            ),
        }

    def select_engine(self, query: str) -> GatewayEngine:
        content = query.lower()
        engines = self.available_engines()

        if any(word in content for word in COMPLEX_WORDS) or len(query) > 120:
            return engines["premium"]
        if any(word in content for word in MEDIUM_WORDS) or len(query) > 40:
            return engines["balanced"]
        if any(word in content for word in SIMPLE_WORDS):
            return engines["fast"]
        return engines["balanced"]

    def resolve_engine(self, requested_model: str, query: str) -> GatewayEngine:
        normalized = requested_model.strip().lower()
        engines = self.available_engines()

        if normalized in ("", "auto"):
            return self.select_engine(query)
        if normalized in engines:
            return engines[normalized]
        raise ValueError(
            "Unsupported model selection. Use 'auto', 'fast', 'balanced', or 'premium'."
        )

    def resolve_model_label(self, engine: GatewayEngine) -> str:
        if engine.name == "balanced" and self.system.query_engine is not None:
            return self._extract_model_name(getattr(self.system.query_engine, "llm", None), fallback="rag-query-engine")
        if engine.name == "premium" and self.system.multi_step_agent is not None:
            return "multi-step-agent"
        return self._extract_model_name(getattr(self.system.llm_twin, "model", None), fallback="fallback-llm")

    def _extract_model_name(self, model: Any, fallback: str) -> str:
        if model is None:
            return fallback
        for attr in ("model_name", "model", "model_id"):
            value = getattr(model, attr, None)
            if isinstance(value, str) and value:
                return value
        return type(model).__name__ or fallback

    def execute(
        self,
        *,
        query: str,
        top_k: int,
        retrieval_mode: str,
        expand_query: bool,
        rewrite_enabled: bool,
        requested_model: str = "auto",
    ) -> GatewayExecutionResult:
        engine = self.resolve_engine(requested_model, query)
        rewritten_query = query
        used_rewrite = False

        if rewrite_enabled and self.system.rewrite_chain is not None:
            candidate = rewrite_question(self.system.rewrite_chain, query)
            if not candidate.startswith("An error occurred") and candidate != "No valid rewritten result":
                rewritten_query = candidate
                used_rewrite = True

        if engine.name == "premium" and self.system.multi_step_agent is not None:
            response = self._run_multi_step(rewritten_query)
            result = GatewayExecutionResult(
                engine=engine,
                query=rewritten_query,
                response=response,
                retrieved_context=[],
                retrieved=[],
                used_rewrite=used_rewrite,
                used_multi_step=True,
            )
            self._record(engine.name)
            return result

        if engine.name in {"balanced", "premium"} and self.system.query_engine is not None:
            result = self._run_query_engine(
                rewritten_query,
                top_k=top_k,
                engine=engine,
                used_rewrite=used_rewrite,
            )
            self._record(engine.name)
            return result

        result = self._run_direct_rag(
            rewritten_query,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            expand_query=expand_query,
            engine=engine,
            used_rewrite=used_rewrite,
        )
        self._record(engine.name)
        return result

    def _run_multi_step(self, query: str) -> str:
        from Inference_pipeline import run_multi_step_search

        return run_multi_step_search(self.system.multi_step_agent, query)

    def _run_query_engine(
        self,
        query: str,
        *,
        top_k: int,
        engine: GatewayEngine,
        used_rewrite: bool,
    ) -> GatewayExecutionResult:
        result = self.system.query_engine.run(
            query,
            retrieve_top_k=top_k,
            rerank_top_n=top_k,
            request_metadata={
                "gateway_engine": engine.name,
                "gateway_tier": engine.tier,
                "gateway_model": self.resolve_model_label(engine),
                "used_rewrite": used_rewrite,
            },
        )
        retrieved = documents_to_retrieval_results(result.documents, top_k=top_k)
        return GatewayExecutionResult(
            engine=engine,
            query=query,
            response=result.answer,
            retrieved_context=[doc.page_content for doc in result.documents],
            retrieved=retrieved,
            used_rewrite=used_rewrite,
            used_multi_step=False,
        )

    def _run_direct_rag(
        self,
        query: str,
        *,
        top_k: int,
        retrieval_mode: str,
        expand_query: bool,
        engine: GatewayEngine,
        used_rewrite: bool,
    ) -> GatewayExecutionResult:
        retrieved = self.system.retrieval_client.retrieve(
            query,
            top_k=top_k,
            mode=retrieval_mode,
            expand_query=expand_query,
        )
        response = self.system.llm_twin.answer(
            query,
            retrieved,
            request_metadata={
                "gateway_engine": engine.name,
                "gateway_tier": engine.tier,
                "gateway_model": self.resolve_model_label(engine),
                "used_rewrite": used_rewrite,
            },
        )
        return GatewayExecutionResult(
            engine=engine,
            query=query,
            response=response,
            retrieved_context=[metadata["text"] for _, metadata in retrieved],
            retrieved=retrieved,
            used_rewrite=used_rewrite,
            used_multi_step=False,
        )

    def _record(self, engine_name: str) -> None:
        self.request_count += 1
        self.engine_counts[engine_name] = self.engine_counts.get(engine_name, 0) + 1

    def stats(self) -> Dict[str, Any]:
        return {
            "engine_types": list(self.available_engines().keys()),
            "request_count": self.request_count,
            "engine_counts": dict(self.engine_counts),
            "query_engine_available": self.system.query_engine is not None,
            "multi_step_available": self.system.multi_step_agent is not None,
            "default_model": self.resolve_model_label(self.select_engine("what is rag")),
        }

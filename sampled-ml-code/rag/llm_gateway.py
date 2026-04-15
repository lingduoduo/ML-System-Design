from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List

from Inference_pipeline import AnswerValidationResult, validate_generated_answer
from feature_pipeline import rewrite_question
from rag_system import CRAGConfig, crag_query_sync, local_docs_sufficiency
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
    selected_model: str
    query: str
    response: str
    retrieved_context: List[str]
    retrieved: List[tuple[float, Dict[str, Any]]] = field(default_factory=list)
    validation: AnswerValidationResult | None = None
    used_rewrite: bool = False
    used_multi_step: bool = False


@dataclass
class LLMGateway:
    system: Any
    request_count: int = 0
    engine_counts: Dict[str, int] = field(default_factory=dict)
    validation_outcomes: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for engine_name in ("fast", "balanced", "premium"):
            self.engine_counts.setdefault(engine_name, 0)
        for outcome in ("grounded", "needs_review"):
            self.validation_outcomes.setdefault(outcome, 0)

    @property
    def validation_counts(self) -> Dict[str, int]:
        return dict(self.validation_outcomes)

    @cached_property
    def engines(self) -> Dict[str, GatewayEngine]:
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

        if any(word in content for word in COMPLEX_WORDS) or len(query) > 120:
            return self.engines["premium"]
        if any(word in content for word in MEDIUM_WORDS) or len(query) > 40:
            return self.engines["balanced"]
        if any(word in content for word in SIMPLE_WORDS):
            return self.engines["fast"]
        return self.engines["balanced"]

    def resolve_engine(self, requested_model: str, query: str) -> GatewayEngine:
        normalized = requested_model.strip().lower()

        if normalized in ("", "auto"):
            return self.select_engine(query)
        if normalized in self.engines:
            return self.engines[normalized]
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

    def _build_request_metadata(
        self,
        engine: GatewayEngine,
        *,
        selected_model: str,
        used_rewrite: bool,
    ) -> Dict[str, Any]:
        return {
            "gateway_engine": engine.name,
            "gateway_tier": engine.tier,
            "gateway_model": selected_model,
            "used_rewrite": used_rewrite,
        }

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
        selected_model = self.resolve_model_label(engine)
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
                selected_model=selected_model,
                query=rewritten_query,
                response=response,
                retrieved_context=[],
                retrieved=[],
                validation=self._validate_response(
                    query=rewritten_query,
                    response=response,
                    retrieved_context=[],
                ),
                used_rewrite=used_rewrite,
                used_multi_step=True,
            )
            self._record(engine.name, result.validation)
            return result

        if engine.name in {"balanced", "premium"} and self.system.query_engine is not None:
            result = self._run_query_engine(
                rewritten_query,
                top_k=top_k,
                engine=engine,
                selected_model=selected_model,
                used_rewrite=used_rewrite,
            )
            self._record(engine.name, result.validation)
            return result

        result = self._run_direct_rag(
            rewritten_query,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            expand_query=expand_query,
            engine=engine,
            selected_model=selected_model,
            used_rewrite=used_rewrite,
        )
        self._record(engine.name, result.validation)
        return result

    def _validate_response(
        self,
        *,
        query: str,
        response: str,
        retrieved_context: List[str],
    ) -> AnswerValidationResult:
        return validate_generated_answer(
            query=query,
            answer=response,
            retrieved_context=retrieved_context,
        )

    def _run_multi_step(self, query: str) -> str:
        from Inference_pipeline import run_multi_step_search

        return run_multi_step_search(self.system.multi_step_agent, query)

    def _run_query_engine(
        self,
        query: str,
        *,
        top_k: int,
        engine: GatewayEngine,
        selected_model: str,
        used_rewrite: bool,
    ) -> GatewayExecutionResult:
        should_use_rag, retrieved_docs, relevance_labels = local_docs_sufficiency(
            query,
            self.system.query_engine.retriever,
            self.system.query_engine.llm,
            CRAGConfig(top_k=top_k),
        )
        if should_use_rag:
            result = self.system.query_engine.run(
                query,
                retrieve_top_k=top_k,
                rerank_top_n=top_k,
                request_metadata=self._build_request_metadata(
                    engine,
                    selected_model=selected_model,
                    used_rewrite=used_rewrite,
                ),
            )
            retrieved = documents_to_retrieval_results(result.documents, top_k=top_k)
            retrieved_context = [doc.page_content for doc in result.documents]
            return GatewayExecutionResult(
                engine=engine,
                selected_model=selected_model,
                query=query,
                response=result.answer,
                retrieved_context=retrieved_context,
                retrieved=retrieved,
                validation=self._validate_response(
                    query=query,
                    response=result.answer,
                    retrieved_context=retrieved_context,
                ),
                used_rewrite=used_rewrite,
                used_multi_step=False,
            )

        crag_retrieved = self._build_crag_retrieval_results(retrieved_docs, relevance_labels, top_k=top_k)
        crag_result = crag_query_sync(
            question=query,
            vectorstore=self.system.query_engine.retriever,
            cfg=CRAGConfig(top_k=top_k),
            llm_answer=self.system.query_engine.llm,
            llm_grader=self.system.query_engine.llm,
            llm_rewrite=self.system.query_engine.llm,
            tavily_tool=None,
            retrieved_docs=retrieved_docs,
            relevance_labels=relevance_labels,
        )
        retrieved_context = crag_result.get("relevant_local", []) + crag_result.get("web_snippets", [])
        return GatewayExecutionResult(
            engine=engine,
            selected_model=selected_model,
            query=query,
            response=crag_result["final_answer"],
            retrieved_context=retrieved_context,
            retrieved=crag_retrieved,
            validation=self._validate_response(
                query=query,
                response=crag_result["final_answer"],
                retrieved_context=retrieved_context,
            ),
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
        selected_model: str,
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
            request_metadata=self._build_request_metadata(
                engine,
                selected_model=selected_model,
                used_rewrite=used_rewrite,
            ),
        )
        retrieved_context = [metadata["text"] for _, metadata in retrieved]
        return GatewayExecutionResult(
            engine=engine,
            selected_model=selected_model,
            query=query,
            response=response,
            retrieved_context=retrieved_context,
            retrieved=retrieved,
            validation=self._validate_response(
                query=query,
                response=response,
                retrieved_context=retrieved_context,
            ),
            used_rewrite=used_rewrite,
            used_multi_step=False,
        )

    def _record(self, engine_name: str, validation: AnswerValidationResult | None = None) -> None:
        self.request_count += 1
        self.engine_counts[engine_name] = self.engine_counts.get(engine_name, 0) + 1
        if validation is not None:
            bucket = "grounded" if validation.grounded else "needs_review"
            self.validation_outcomes[bucket] = self.validation_outcomes.get(bucket, 0) + 1

    def _build_crag_retrieval_results(
        self,
        retrieved_docs: List[Any],
        relevance_labels: List[str],
        *,
        top_k: int,
    ) -> List[tuple[float, Dict[str, Any]]]:
        retrieved = documents_to_retrieval_results(retrieved_docs, top_k=top_k)
        if not relevance_labels:
            return retrieved

        enriched: List[tuple[float, Dict[str, Any]]] = []
        for index, (score, metadata) in enumerate(retrieved):
            updated_metadata = dict(metadata)
            if index < len(relevance_labels):
                updated_metadata["crag_relevance"] = relevance_labels[index]
            enriched.append((score, updated_metadata))
        return enriched

    def stats(self) -> Dict[str, Any]:
        return {
            "engine_types": list(self.engines.keys()),
            "request_count": self.request_count,
            "engine_counts": dict(self.engine_counts),
            "validation_counts": self.validation_counts,
            "query_engine_available": self.system.query_engine is not None,
            "multi_step_available": self.system.multi_step_agent is not None,
            "default_model": self.resolve_model_label(self.select_engine("what is rag")),
        }

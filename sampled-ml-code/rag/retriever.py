from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable, Dict, List, Tuple

from feature_pipeline import RetrievalMode, RetrieverIndex, SimpleEmbedder, VectorDB

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict, Field

    LANGCHAIN_RETRIEVER_SUPPORT = True
except ImportError:  # pragma: no cover - optional dependency path
    Document = Any  # type: ignore[assignment]
    BaseRetriever = object  # type: ignore[assignment]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    LANGCHAIN_RETRIEVER_SUPPORT = False


@dataclass
class RetrievalClient:
    embedder: SimpleEmbedder
    vector_db: VectorDB
    index: RetrieverIndex | None = None

    def __post_init__(self) -> None:
        self.index = RetrieverIndex.from_components(self.embedder, self.vector_db)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        mode: RetrievalMode = "dense",
        expand_query: bool = False,
    ) -> List[Tuple[float, Dict]]:
        assert self.index is not None
        return self.index.search(query, top_k=top_k, mode=mode, expand_query=expand_query)


@dataclass
class LangChainRetrievalClient:
    retriever: Any

    def _configure_top_k(self, top_k: int) -> None:
        if top_k <= 0:
            return

        if hasattr(self.retriever, "search_kwargs") and isinstance(self.retriever.search_kwargs, dict):
            self.retriever.search_kwargs["k"] = top_k
        if hasattr(self.retriever, "k"):
            try:
                self.retriever.k = top_k
            except Exception:
                pass
        if hasattr(self.retriever, "vector_top_k"):
            try:
                self.retriever.vector_top_k = top_k
            except Exception:
                pass
        if hasattr(self.retriever, "bm25_top_k"):
            try:
                self.retriever.bm25_top_k = top_k
            except Exception:
                pass
        if hasattr(self.retriever, "runtime_top_k"):
            try:
                self.retriever.runtime_top_k = top_k
            except Exception:
                pass

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        mode: RetrievalMode = "dense",
        expand_query: bool = False,
    ) -> List[Tuple[float, Dict]]:
        del mode, expand_query
        self._configure_top_k(top_k)
        documents = self.retriever.invoke(query)
        if top_k > 0:
            documents = documents[:top_k]
        results: List[Tuple[float, Dict]] = []
        for index, doc in enumerate(documents):
            metadata = dict(getattr(doc, "metadata", {}))
            metadata.setdefault("chunk_id", index)
            metadata["text"] = doc.page_content
            score = float(metadata.get("rerank_score", metadata.get("fusion_score", 1.0)))
            results.append((score, metadata))
        return results


def build_llm_reranker(llm: Any) -> Callable[[str, Any], float]:
    """Create a simple LLM-based reranker that scores query-document relevance from 0 to 100."""

    def _extract_numeric_score(raw: Any) -> float:
        text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
        match = re.search(r"\b100\b|\b\d{1,2}\b", text)
        if match is None:
            return 0.0
        return float(max(0, min(100, int(match.group(0)))))

    def rerank(query: str, doc: Any) -> float:
        prompt = (
            "Evaluate how relevant the following document is to the user's query. "
            "Return only one integer from 0 to 100, where 100 means highly relevant.\n\n"
            f"User query: {query}\n\n"
            f"Document:\n{doc.page_content}\n\n"
            "Relevance score:"
        )
        try:
            if hasattr(llm, "invoke"):
                return _extract_numeric_score(llm.invoke(prompt))
            if hasattr(llm, "generate"):
                return _extract_numeric_score(llm.generate(prompt))
        except Exception:
            return 0.0
        return 0.0

    return rerank


@dataclass
class RetrievalPath:
    name: str
    weight: float
    retrieve: Callable[[str], List[Tuple[Document, float]]]

    def run(self, query: str) -> List[Tuple[Document, float, str]]:
        return [
            (doc, raw_score * self.weight, self.name)
            for doc, raw_score in self.retrieve(query)
        ]


@dataclass
class DynamicScoreFilter:
    min_results: int = 1

    def select(self, candidates: List[Tuple[Document, float, str]]) -> List[Tuple[Document, float, str]]:
        if not candidates:
            return []

        scores = [score for _, score, _ in candidates]
        mean_score = sum(scores) / len(scores)
        std_dev = math.sqrt(sum((score - mean_score) ** 2 for score in scores) / len(scores))
        threshold = mean_score + std_dev

        ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
        kept = [item for item in ranked if item[1] >= threshold]
        if kept:
            return kept
        return ranked[: max(1, self.min_results)]


if LANGCHAIN_RETRIEVER_SUPPORT:
    class MultiPathRetriever(BaseRetriever):
        """Hybrid retriever with multiple recall paths, fusion, dynamic filtering, and reranking."""

        bm25_retriever: Any
        vectorstore: Any
        vector_top_k: int = 2
        bm25_top_k: int = 2
        vector_weight: float = 0.8
        bm25_weight: float = 0.6
        reranker: Callable[[str, Document], float] | None = None
        rerank_top_k: int | None = None
        runtime_top_k: int | None = None
        score_filter: DynamicScoreFilter = Field(default_factory=DynamicScoreFilter)

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def _get_relevant_documents(self, query: str) -> List[Document]:
            """Run multiple retrievers, fuse results, dedupe, filter, and optionally rerank."""
            merged = self._merge_and_deduplicate(self._collect_candidates(query))
            if not merged:
                return []

            kept = self._apply_dynamic_threshold(merged)
            if self.reranker is not None:
                kept = self._rerank(query, kept)
            kept = self._limit_results(kept)

            output: List[Document] = []
            for doc, score, source in kept:
                metadata = dict(doc.metadata or {})
                metadata.setdefault("fusion_score", float(score))
                metadata["final_score"] = float(score)
                metadata["retrieval_path"] = source
                output.append(Document(page_content=doc.page_content, metadata=metadata))
            return output

        def _collect_candidates(self, query: str) -> List[Tuple[Document, float, str]]:
            candidates: List[Tuple[Document, float, str]] = []
            for path in self._build_paths():
                candidates.extend(path.run(query))
            return candidates

        def _build_paths(self) -> List[RetrievalPath]:
            return [
                RetrievalPath(
                    name="bm25",
                    weight=self.bm25_weight,
                    retrieve=self._retrieve_bm25,
                ),
                RetrievalPath(
                    name="vector",
                    weight=self.vector_weight,
                    retrieve=self._retrieve_vector,
                ),
            ]

        def _retrieve_bm25(self, query: str) -> List[Tuple[Document, float]]:
            try:
                self.bm25_retriever.k = self.bm25_top_k
            except Exception:
                pass

            bm25_docs = self.bm25_retriever.invoke(query)
            return [
                (doc, 1.0 / (rank + 1))
                for rank, doc in enumerate(bm25_docs)
            ]

        def _retrieve_vector(self, query: str) -> List[Tuple[Document, float]]:
            vector_hits: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_score(
                query,
                k=self.vector_top_k,
            )
            return [
                (doc, 1.0 / (1.0 + distance))
                for doc, distance in vector_hits
            ]

        def _merge_and_deduplicate(
            self,
            candidates: List[Tuple[Document, float, str]],
        ) -> List[Tuple[Document, float, str]]:
            best_by_content: Dict[str, Tuple[Document, float, str]] = {}
            for doc, score, source in candidates:
                key = doc.page_content
                current = best_by_content.get(key)
                if current is None or score > current[1]:
                    best_by_content[key] = (doc, score, source)
            return list(best_by_content.values())

        def _apply_dynamic_threshold(
            self,
            candidates: List[Tuple[Document, float, str]],
        ) -> List[Tuple[Document, float, str]]:
            return self.score_filter.select(candidates)

        def _rerank(
            self,
            query: str,
            candidates: List[Tuple[Document, float, str]],
        ) -> List[Tuple[Document, float, str]]:
            reranked: List[Tuple[Document, float, str]] = []
            for doc, score, source in candidates:
                rerank_score = float(self.reranker(query, doc))
                metadata = dict(doc.metadata or {})
                metadata["fusion_score"] = float(score)
                metadata["rerank_score"] = rerank_score
                reranked.append(
                    (
                        Document(page_content=doc.page_content, metadata=metadata),
                        rerank_score,
                        source,
                    )
                )

            reranked.sort(key=lambda item: item[1], reverse=True)
            if self.rerank_top_k is not None and self.rerank_top_k > 0:
                return reranked[:self.rerank_top_k]
            return reranked

        def _limit_results(
            self,
            candidates: List[Tuple[Document, float, str]],
        ) -> List[Tuple[Document, float, str]]:
            if self.runtime_top_k is not None and self.runtime_top_k > 0:
                return candidates[:self.runtime_top_k]
            return candidates
else:
    class MultiPathRetriever:  # pragma: no cover - optional dependency path
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise ImportError(
                "MultiPathRetriever requires `langchain-core` and `pydantic`. "
                "When available, it runs multiple retrievers, then fuses, deduplicates, "
                "dynamically filters, and optionally reranks the results."
            )

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Tuple

from feature_pipeline import RetrievalMode, RetrieverIndex, SimpleEmbedder, VectorDB

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict

    LANGCHAIN_RETRIEVER_SUPPORT = True
except ImportError:  # pragma: no cover - optional dependency path
    Document = Any  # type: ignore[assignment]
    BaseRetriever = object  # type: ignore[assignment]
    ConfigDict = None  # type: ignore[assignment]
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


if LANGCHAIN_RETRIEVER_SUPPORT:
    class MultiPathRetriever(BaseRetriever):
        """Hybrid retriever that fuses lexical and vector paths, then optionally reranks."""

        bm25_retriever: Any
        vectorstore: Any
        vector_top_k: int = 2
        bm25_top_k: int = 2
        vector_weight: float = 0.8
        bm25_weight: float = 0.6
        reranker: Callable[[str, Document], float] | None = None
        rerank_top_k: int | None = None
        runtime_top_k: int | None = None

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def _get_relevant_documents(self, query: str) -> List[Document]:
            """Fuse BM25 and vector retrieval, then keep the strongest candidates."""
            try:
                self.bm25_retriever.k = self.bm25_top_k
            except Exception:
                pass

            bm25_docs = self.bm25_retriever.invoke(query)
            bm25_scored = [
                (doc, (1.0 / (rank + 1)) * self.bm25_weight, "bm25")
                for rank, doc in enumerate(bm25_docs)
            ]

            vector_hits: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_score(
                query,
                k=self.vector_top_k,
            )
            vector_scored = [
                (doc, (1.0 / (1.0 + distance)) * self.vector_weight, "vector")
                for doc, distance in vector_hits
            ]

            merged = self._merge_and_deduplicate(vector_scored + bm25_scored)
            if not merged:
                return []

            kept = self._apply_dynamic_threshold(merged)
            if self.reranker is not None:
                kept = self._rerank(query, kept)
            kept = self._limit_results(kept)

            output: List[Document] = []
            for doc, score, source in kept:
                metadata = dict(doc.metadata or {})
                metadata["fusion_score"] = float(score)
                metadata["retrieval_path"] = source
                output.append(Document(page_content=doc.page_content, metadata=metadata))
            return output

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
            scores = [score for _, score, _ in candidates]
            mean_score = sum(scores) / len(scores)
            std_dev = math.sqrt(sum((score - mean_score) ** 2 for score in scores) / len(scores))
            threshold = mean_score + std_dev

            ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
            kept = [item for item in ranked if item[1] >= threshold] or ranked[:1]
            return kept

        def _rerank(
            self,
            query: str,
            candidates: List[Tuple[Document, float, str]],
        ) -> List[Tuple[Document, float, str]]:
            reranked: List[Tuple[Document, float, str]] = []
            for doc, score, source in candidates:
                rerank_score = float(self.reranker(query, doc))
                metadata = dict(doc.metadata or {})
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
                "MultiPathRetriever requires `langchain-core` and `pydantic`."
            )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from feature_pipeline import RetrievalMode, RetrieverIndex, SimpleEmbedder, VectorDB


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

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        mode: RetrievalMode = "dense",
        expand_query: bool = False,
    ) -> List[Tuple[float, Dict]]:
        del top_k, mode, expand_query
        documents = self.retriever.invoke(query)
        results: List[Tuple[float, Dict]] = []
        for index, doc in enumerate(documents):
            metadata = dict(getattr(doc, "metadata", {}))
            metadata.setdefault("chunk_id", index)
            metadata["text"] = doc.page_content
            results.append((1.0, metadata))
        return results

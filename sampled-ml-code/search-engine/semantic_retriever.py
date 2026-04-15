from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import heapq
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from document_processor import TextProcessor
from vocabulary import Vocabulary


@dataclass
class RetrievedDocument:
    """Lightweight retrieval result similar to a LangChain document."""

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexedSemanticDocument:
    """Internal indexed representation used for efficient sparse semantic lookup."""

    document: RetrievedDocument
    vector: Dict[str, float]
    dataset_id: str
    document_enabled: bool
    segment_enabled: bool


class InMemorySemanticVectorStore:
    """Small in-memory semantic store with dataset-aware filtering."""

    def __init__(self, text_processor: Optional[TextProcessor] = None):
        self.text_processor = text_processor or TextProcessor()
        self.vocab = Vocabulary()
        self.documents: List[IndexedSemanticDocument] = []
        self.term_to_doc_ids: Dict[str, Set[int]] = defaultdict(set)
        self.dataset_to_doc_ids: Dict[str, Set[int]] = defaultdict(set)
        self.enabled_doc_ids: Set[int] = set()
        self.enabled_segment_ids: Set[int] = set()

    def clear(self) -> None:
        self.documents = []
        self.term_to_doc_ids = defaultdict(set)
        self.dataset_to_doc_ids = defaultdict(set)
        self.enabled_doc_ids = set()
        self.enabled_segment_ids = set()
        self.vocab = Vocabulary()

    def _tokenize(self, text: str) -> List[str]:
        return self.text_processor.tokenize(text)

    def _embed_from_tokens(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}

        counts: Dict[str, float] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0.0) + 1.0

        norm = float(np.sqrt(sum(value * value for value in counts.values())))
        if norm == 0.0:
            return counts
        return {token: value / norm for token, value in counts.items()}

    def _embed(self, text: str) -> Dict[str, float]:
        return self._embed_from_tokens(self._tokenize(text))

    def _cosine_similarity(self, left: Dict[str, float], right: Dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(term, 0.0) for term, value in left.items())

    def add_documents(self, documents: Sequence[RetrievedDocument], reset: bool = False) -> None:
        if reset:
            self.clear()

        for document in documents:
            metadata = document.metadata
            tokens = self._tokenize(document.page_content)
            vector = self._embed_from_tokens(tokens)
            dataset_id = str(metadata.get("dataset_id", ""))
            doc_id = len(self.documents)

            indexed_document = IndexedSemanticDocument(
                document=document,
                vector=vector,
                dataset_id=dataset_id,
                document_enabled=bool(metadata.get("document_enabled", True)),
                segment_enabled=bool(metadata.get("segment_enabled", True)),
            )
            self.documents.append(indexed_document)

            self.dataset_to_doc_ids[dataset_id].add(doc_id)
            if indexed_document.document_enabled:
                self.enabled_doc_ids.add(doc_id)
            if indexed_document.segment_enabled:
                self.enabled_segment_ids.add(doc_id)
            for term in vector:
                self.term_to_doc_ids[term].add(doc_id)

            self.vocab.add_many(tokens)

    def _candidate_doc_ids(self, query_vector: Dict[str, float], filters: Dict[str, Any]) -> Set[int]:
        candidate_ids: Set[int] = set()
        for term in query_vector:
            candidate_ids.update(self.term_to_doc_ids.get(term, set()))

        if not candidate_ids:
            return set()

        allowed_dataset_ids = {str(item) for item in filters.get("dataset_ids", [])}
        require_document_enabled = filters.get("document_enabled", True)
        require_segment_enabled = filters.get("segment_enabled", True)

        if allowed_dataset_ids:
            dataset_matches: Set[int] = set()
            for dataset_id in allowed_dataset_ids:
                dataset_matches.update(self.dataset_to_doc_ids.get(dataset_id, set()))
            candidate_ids &= dataset_matches

        if require_document_enabled:
            candidate_ids &= self.enabled_doc_ids
        if require_segment_enabled:
            candidate_ids &= self.enabled_segment_ids

        return candidate_ids

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> List[Tuple[RetrievedDocument, float]]:
        query_vector = self._embed(query)
        if not query_vector:
            return []

        filters = filters or {}
        candidate_ids = self._candidate_doc_ids(query_vector, filters)
        if not candidate_ids:
            return []

        scored_documents = []
        for doc_id in candidate_ids:
            indexed_document = self.documents[doc_id]
            score = self._cosine_similarity(query_vector, indexed_document.vector)
            if score > 0.0:
                scored_documents.append((indexed_document.document, score))

        if not scored_documents:
            return []
        return heapq.nlargest(k, scored_documents, key=lambda item: item[1])


class SemanticRetriever:
    """Semantic retriever with pluggable vector store and in-memory fallback."""

    def __init__(
        self,
        dataset_ids: Optional[Iterable[str]] = None,
        vector_store: Optional[Any] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
        text_processor: Optional[TextProcessor] = None,
    ):
        self.dataset_ids = [str(dataset_id) for dataset_id in (dataset_ids or [])]
        self.search_kwargs = dict(search_kwargs or {})
        self.text_processor = text_processor or TextProcessor()
        self.vector_store = vector_store or InMemorySemanticVectorStore(self.text_processor)

    def index_documents(
        self,
        documents: Sequence[Dict[str, Any]],
        content_field: str = "content",
        reset: bool = True,
    ) -> None:
        """Index plain dictionaries into the semantic vector store."""
        retrievable_documents = []
        for document in documents:
            content = str(document.get(content_field) or self.text_processor.normalize_document(document))
            metadata = document.copy()
            metadata.setdefault("dataset_id", "default")
            metadata.setdefault("document_enabled", True)
            metadata.setdefault("segment_enabled", True)
            retrievable_documents.append(RetrievedDocument(page_content=content, metadata=metadata))

        if hasattr(self.vector_store, "add_documents"):
            self.vector_store.add_documents(retrievable_documents, reset=reset)
        else:
            raise TypeError("The configured vector_store does not support add_documents().")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """Run semantic similarity search and attach scores to metadata."""
        search_kwargs = dict(self.search_kwargs)
        default_k = search_kwargs.pop("k", 4)
        k = top_k if top_k is not None else default_k

        filters = search_kwargs.pop("filters", {})
        if isinstance(filters, dict):
            filters = {
                **filters,
                "dataset_ids": self.dataset_ids or filters.get("dataset_ids", []),
                "document_enabled": filters.get("document_enabled", True),
                "segment_enabled": filters.get("segment_enabled", True),
            }

        search_result = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filters=filters,
            **search_kwargs,
        )
        if not search_result:
            return []

        retrieved_documents = []
        for rank, (document, score) in enumerate(search_result, start=1):
            metadata = dict(document.metadata)
            metadata["score"] = float(score)
            metadata["rank"] = rank
            retrieved_documents.append(
                RetrievedDocument(page_content=document.page_content, metadata=metadata)
            )

        return retrieved_documents


if __name__ == "__main__":
    sample_documents = [
        {
            "dataset_id": "spa-dataset",
            "document_id": "doc-1",
            "segment_id": "seg-1",
            "content": "Shunjing Hot Spring offers outdoor pools and scenic mountain views.",
            "document_enabled": True,
            "segment_enabled": True,
        },
        {
            "dataset_id": "spa-dataset",
            "document_id": "doc-2",
            "segment_id": "seg-2",
            "content": "Jiuhua Resort has spa services, family amenities, and relaxing hot springs.",
            "document_enabled": True,
            "segment_enabled": True,
        },
        {
            "dataset_id": "city-dataset",
            "document_id": "doc-3",
            "segment_id": "seg-3",
            "content": "City museum tickets and exhibition schedules for downtown visitors.",
            "document_enabled": True,
            "segment_enabled": True,
        },
    ]

    retriever = SemanticRetriever(dataset_ids=["spa-dataset"], search_kwargs={"k": 2})
    retriever.index_documents(sample_documents)

    results = retriever.retrieve("best spa hot spring resort")
    for document in results:
        print(
            f"{document.metadata['rank']}. "
            f"{document.metadata['document_id']} "
            f"(score={document.metadata['score']:.4f}) "
            f"- {document.page_content}"
        )

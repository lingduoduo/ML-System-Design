import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from document_processor import TextProcessor
from semantic_retriever import RetrievedDocument, SemanticRetriever


class SemanticIndexBuilder:
    """Prepare cleaned semantic-search documents from text or JSON inputs."""

    def __init__(self, text_processor: Optional[TextProcessor] = None):
        self.text_processor = text_processor or TextProcessor()
        self.documents: List[Dict[str, Any]] = []

    def clear(self) -> None:
        self.documents = []

    def add_document(self, source: Any, doc_id: str) -> None:
        content = self.text_processor.normalize_document(source)
        if not content:
            return

        self.documents.append(
            {
                "dataset_id": "semantic-index",
                "document_id": doc_id,
                "segment_id": doc_id,
                "node_id": doc_id,
                "content": content,
                "type": "semantic_document",
                "document_enabled": True,
                "segment_enabled": True,
            }
        )

    def get_statistics(self) -> Dict[str, Any]:
        lengths = [len(document["content"].split()) for document in self.documents]
        return {
            "total_documents": len(self.documents),
            "avg_document_length": float(np.mean(lengths)) if lengths else 0.0,
            "max_document_length": max(lengths) if lengths else 0,
        }


class SemanticRetrievalPipeline:
    """End-to-end semantic retrieval pipeline."""

    def __init__(self, text_processor: Optional[TextProcessor] = None):
        self.text_processor = text_processor or TextProcessor()
        self.index_builder = SemanticIndexBuilder(text_processor=self.text_processor)
        self.retriever = SemanticRetriever(
            dataset_ids=["semantic-index"],
            search_kwargs={"k": 10},
            text_processor=self.text_processor,
        )
        self.is_built = False

    async def build_index(self, documents: List[Tuple[Any, str]], chunk_size: int = 512) -> None:
        """Build a semantic index from cleaned documents."""
        del chunk_size

        print("Step 1: Normalizing semantic documents...")
        self.index_builder.clear()
        for source, doc_id in documents:
            self.index_builder.add_document(source, doc_id)

        stats = self.index_builder.get_statistics()
        print(f"Prepared {stats['total_documents']} semantic documents")
        print(f"Average document length: {stats['avg_document_length']:.2f}")
        print(f"Max document length: {stats['max_document_length']}")

        print("Step 2: Indexing semantic documents...")
        self.retriever.index_documents(self.index_builder.documents, reset=True)
        self.is_built = True
        print("Semantic index building complete!")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index first.")

        results = self.retriever.retrieve(query, top_k=top_k)
        return [self._to_result(document) for document in results]

    def _to_result(self, document: RetrievedDocument) -> Dict[str, Any]:
        metadata = dict(document.metadata)
        return {
            "id": metadata.get("segment_id", metadata.get("document_id")),
            "text": document.page_content,
            "score": float(metadata.get("score", 0.0)),
            "type": metadata.get("type", "semantic_document"),
            "metadata": metadata,
        }

    def get_index_stats(self) -> Dict[str, Any]:
        return self.index_builder.get_statistics()


class PerformanceEvaluator:
    """Evaluate semantic retrieval against document-level ground truth."""

    def __init__(self):
        self.metrics = {}

    def evaluate_retrieval(
        self,
        queries: List[str],
        ground_truth: List[List[str]],
        pipeline: SemanticRetrievalPipeline,
    ) -> Dict[str, float]:
        precision_scores = []
        recall_scores = []

        for query, gt_docs in zip(queries, ground_truth):
            results = pipeline.retrieve(query, top_k=10)
            retrieved_ids = {result["metadata"].get("document_id", result["id"]) for result in results}
            gt_set = set(gt_docs)

            precision = len(gt_set & retrieved_ids) / len(retrieved_ids) if retrieved_ids else 0.0
            recall = len(gt_set & retrieved_ids) / len(gt_set) if gt_set else 1.0

            precision_scores.append(precision)
            recall_scores.append(recall)

        avg_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
        avg_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
        avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-8)

        return {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
        }


if __name__ == "__main__":
    async def main():
        documents = [
            (
                "New Jersey has several excellent hot springs. Shunjing Hot Spring is well-known for its resort facilities. Jiuhua Mountain Resort offers scenic spa services. The Capital Hot Spring provides popular hotel amenities.",
                "doc1",
            ),
            (
                "To reach the spa resort, take the main highway and follow the signs. The resort is located in a beautiful mountain area with easy access from downtown.",
                "doc2",
            ),
            (
                "This hot spring is famous because it has unique mineral content that provides therapeutic benefits. Unlike regular spas, it offers authentic hot spring experience with natural geothermal water.",
                "doc3",
            ),
        ]

        pipeline = SemanticRetrievalPipeline()
        await pipeline.build_index(documents)

        queries = [
            "What are the best hot springs in New Jersey?",
            "How do I get to the spa resort?",
            "Why is this hot spring famous?",
        ]

        print("\n=== Semantic Retrieval Results ===")
        for query in queries:
            print(f"\nQuery: {query}")
            results = pipeline.retrieve(query, top_k=3)
            for i, result in enumerate(results, start=1):
                print(f"{i}. [{result['type']}] {result['text'][:100]}... (score: {result['score']:.3f})")

        ground_truth = [["doc1"], ["doc2"], ["doc3"]]
        evaluator = PerformanceEvaluator()
        metrics = evaluator.evaluate_retrieval(queries, ground_truth, pipeline)

        print("\n=== Performance Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")

    asyncio.run(main())

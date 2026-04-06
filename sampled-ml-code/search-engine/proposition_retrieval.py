"""Backward-compatible shim for the renamed semantic retrieval pipeline."""

from semantic_retriever import RetrievedDocument, SemanticRetriever
from semantic_pipeline import PerformanceEvaluator, SemanticIndexBuilder, SemanticRetrievalPipeline

# Older examples imported this name directly; keep it available while the repo
# transitions to semantic-first naming.
PropositionRetrievalPipeline = SemanticRetrievalPipeline

__all__ = [
    "PerformanceEvaluator",
    "PropositionRetrievalPipeline",
    "RetrievedDocument",
    "SemanticIndexBuilder",
    "SemanticRetrievalPipeline",
    "SemanticRetriever",
]

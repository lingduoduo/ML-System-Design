import asyncio
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import re

try:
    from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.retrievers import RecursiveRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.schema import BaseNode, TextNode, NodeRelationship, RelatedNodeInfo
    from llama_index.core.llms import LLM
    from llama_index.core.embeddings import BaseEmbedding
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    print("LlamaIndex not available. Install with: pip install llama-index")


class PropositionNode:
    """Represents a proposition extracted from text."""

    def __init__(self, text: str, parent_node_id: str, proposition_id: str, confidence: float = 1.0):
        self.text = text
        self.parent_node_id = parent_node_id
        self.proposition_id = proposition_id
        self.confidence = confidence
        self.embedding = None
        self.metadata = {
            "type": "proposition",
            "parent_id": parent_node_id,
            "confidence": confidence
        }


class TextNode:
    """Represents a basic text chunk/node."""

    def __init__(self, text: str, node_id: str, doc_id: str = None):
        self.text = text
        self.node_id = node_id
        self.doc_id = doc_id or node_id
        self.propositions: List[PropositionNode] = []
        self.embedding = None
        self.metadata = {
            "type": "node",
            "doc_id": doc_id,
            "proposition_count": 0
        }


class PropositionExtractor:
    """Extracts propositions from text using LLM."""

    def __init__(self, llm: Optional[Any] = None, proposition_prompt: str = None):
        self.llm = llm
        self.proposition_prompt = proposition_prompt or self._default_proposition_prompt()

    def _default_proposition_prompt(self) -> str:
        return """
        Extract the key propositions from the following text. A proposition is a standalone factual claim or statement that can be true or false.

        Guidelines:
        - Break down complex sentences into simple, atomic propositions
        - Each proposition should be a complete, self-contained statement
        - Focus on factual claims, not opinions or questions
        - Preserve the original meaning and context
        - Number each proposition

        Text: {text}

        Propositions:
        """

    async def extract_propositions(self, text: str, node_id: str) -> List[PropositionNode]:
        """Extract propositions from text asynchronously."""
        if not self.llm:
            # Fallback: simple sentence splitting
            return self._extract_propositions_fallback(text, node_id)

        try:
            prompt = self.proposition_prompt.format(text=text)
            response = await self.llm.acomplete(prompt)

            propositions = []
            lines = str(response).strip().split('\n')

            for i, line in enumerate(lines):
                line = line.strip()
                if line and not line.startswith(('Text:', 'Propositions:', '-')):
                    # Clean up numbering
                    line = re.sub(r'^\d+\.?\s*', '', line)
                    if line:
                        prop_id = f"{node_id}_prop_{i}"
                        proposition = PropositionNode(line, node_id, prop_id)
                        propositions.append(proposition)

            return propositions

        except Exception as e:
            print(f"LLM extraction failed: {e}. Using fallback.")
            return self._extract_propositions_fallback(text, node_id)

    def _extract_propositions_fallback(self, text: str, node_id: str) -> List[PropositionNode]:
        """Fallback proposition extraction using sentence splitting."""
        # Simple sentence-based extraction
        sentences = re.split(r'[.!?]+', text)
        propositions = []

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter very short sentences
                prop_id = f"{node_id}_prop_{i}"
                proposition = PropositionNode(sentence, node_id, prop_id, confidence=0.7)
                propositions.append(proposition)

        return propositions


class HybridIndexBuilder:
    """Builds hybrid index with nodes and propositions."""

    def __init__(self, embedding_model: Optional[Any] = None):
        self.embedding_model = embedding_model
        self.nodes: Dict[str, TextNode] = {}
        self.propositions: Dict[str, PropositionNode] = {}
        self.node_proposition_map: Dict[str, List[str]] = defaultdict(list)

        if LLAMA_INDEX_AVAILABLE:
            self.vector_index = None
        else:
            self.vector_index = None

    def add_document(self, text: str, doc_id: str, chunk_size: int = 512, chunk_overlap: int = 50):
        """Add document and create nodes."""
        if LLAMA_INDEX_AVAILABLE:
            # Use LlamaIndex SentenceSplitter
            splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            nodes = splitter.split_text(text)

            for i, node_text in enumerate(nodes):
                node_id = f"{doc_id}_node_{i}"
                text_node = TextNode(node_text, node_id, doc_id)
                self.nodes[node_id] = text_node
        else:
            # Fallback: simple splitting
            sentences = re.split(r'[.!?]+', text)
            current_chunk = []
            current_length = 0
            chunk_idx = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sentence_length = len(sentence)

                if current_length + sentence_length > chunk_size and current_chunk:
                    # Create node
                    node_text = ' '.join(current_chunk)
                    node_id = f"{doc_id}_node_{chunk_idx}"
                    text_node = TextNode(node_text, node_id, doc_id)
                    self.nodes[node_id] = text_node

                    current_chunk = [sentence]
                    current_length = sentence_length
                    chunk_idx += 1
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length

            # Add final chunk
            if current_chunk:
                node_text = ' '.join(current_chunk)
                node_id = f"{doc_id}_node_{chunk_idx}"
                text_node = TextNode(node_text, node_id, doc_id)
                self.nodes[node_id] = text_node

    async def extract_propositions(self, extractor: PropositionExtractor):
        """Extract propositions from all nodes asynchronously."""
        tasks = []
        for node_id, node in self.nodes.items():
            task = extractor.extract_propositions(node.text, node_id)
            tasks.append(task)

        # Execute all extractions concurrently
        proposition_lists = await asyncio.gather(*tasks)

        # Store propositions
        for node_id, propositions in zip(self.nodes.keys(), proposition_lists):
            self.nodes[node_id].propositions = propositions
            self.nodes[node_id].metadata["proposition_count"] = len(propositions)

            for prop in propositions:
                self.propositions[prop.proposition_id] = prop
                self.node_proposition_map[node_id].append(prop.proposition_id)

    def build_vector_index(self):
        """Build vector index with both nodes and propositions."""
        if not LLAMA_INDEX_AVAILABLE:
            print("LlamaIndex not available. Cannot build vector index.")
            return None

        # Create LlamaIndex documents
        llama_nodes = []

        # Add original nodes
        for node_id, node in self.nodes.items():
            llama_node = Document(
                text=node.text,
                id_=node_id,
                metadata={
                    **node.metadata,
                    "node_type": "original",
                    "has_propositions": len(node.propositions) > 0
                }
            )
            llama_nodes.append(llama_node)

        # Add proposition nodes
        for prop_id, proposition in self.propositions.items():
            llama_node = Document(
                text=proposition.text,
                id_=prop_id,
                metadata={
                    **proposition.metadata,
                    "node_type": "proposition",
                    "parent_node_id": proposition.parent_node_id
                }
            )
            llama_nodes.append(llama_node)

        # Build vector index
        self.vector_index = VectorStoreIndex(llama_nodes, embed_model=self.embedding_model)
        return self.vector_index

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_nodes": len(self.nodes),
            "total_propositions": len(self.propositions),
            "avg_propositions_per_node": len(self.propositions) / max(1, len(self.nodes)),
            "nodes_with_propositions": sum(1 for node in self.nodes.values() if node.propositions),
            "proposition_confidence_avg": np.mean([p.confidence for p in self.propositions.values()]) if self.propositions else 0.0
        }


class RecursivePropositionRetriever:
    """Recursive retriever that prioritizes propositions but falls back to nodes."""

    def __init__(self, index_builder: HybridIndexBuilder, similarity_threshold: float = 0.7):
        self.index_builder = index_builder
        self.similarity_threshold = similarity_threshold

        if LLAMA_INDEX_AVAILABLE and index_builder.vector_index:
            # Create retrievers
            base_retriever = index_builder.vector_index.as_retriever(similarity_top_k=10)

            # Custom recursive retriever
            self.retriever = RecursiveRetriever(
                root_retriever=base_retriever,
                child_retrievers={},
                recursive_retriever_fn=self._recursive_retrieve_fn
            )
        else:
            self.retriever = None

    def _recursive_retrieve_fn(self, nodes: List[BaseNode]) -> Dict[str, Any]:
        """Custom recursive retrieval function."""
        proposition_nodes = []
        original_nodes = []

        for node in nodes:
            if node.metadata.get("node_type") == "proposition":
                proposition_nodes.append(node)
            else:
                original_nodes.append(node)

        # If we have propositions, prefer them
        if proposition_nodes:
            return {"nodes": proposition_nodes[:5]}  # Top propositions

        # Otherwise, return original nodes
        return {"nodes": original_nodes[:3]}

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve with proposition-first, node-fallback strategy."""
        if not self.retriever:
            return self._fallback_retrieve(query, top_k)

        try:
            # Use LlamaIndex retriever
            results = self.retriever.retrieve(query)

            retrieved_items = []
            for result in results[:top_k]:
                item = {
                    "id": result.id_,
                    "text": result.text,
                    "score": getattr(result, 'score', 1.0),
                    "type": result.metadata.get("node_type", "unknown"),
                    "metadata": result.metadata
                }
                retrieved_items.append(item)

            return retrieved_items

        except Exception as e:
            print(f"LlamaIndex retrieval failed: {e}. Using fallback.")
            return self._fallback_retrieve(query, top_k)

    def _fallback_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback retrieval using simple text matching."""
        query_terms = set(query.lower().split())
        scored_items = []

        # Score propositions
        for prop_id, proposition in self.index_builder.propositions.items():
            prop_terms = set(proposition.text.lower().split())
            overlap = len(query_terms & prop_terms)
            score = overlap / max(len(query_terms), 1)

            if score > 0:
                scored_items.append({
                    "id": prop_id,
                    "text": proposition.text,
                    "score": score,
                    "type": "proposition",
                    "metadata": proposition.metadata
                })

        # Score nodes if no good propositions
        if len(scored_items) < top_k:
            for node_id, node in self.index_builder.nodes.items():
                node_terms = set(node.text.lower().split())
                overlap = len(query_terms & node_terms)
                score = overlap / max(len(query_terms), 1) * 0.8  # Slightly lower weight

                if score > 0:
                    scored_items.append({
                        "id": node_id,
                        "text": node.text,
                        "score": score,
                        "type": "node",
                        "metadata": node.metadata
                    })

        # Sort and return top results
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        return scored_items[:top_k]


class PropositionRetrievalPipeline:
    """Complete proposition-based retrieval pipeline."""

    def __init__(self, llm: Optional[Any] = None, embedding_model: Optional[Any] = None):
        self.extractor = PropositionExtractor(llm)
        self.index_builder = HybridIndexBuilder(embedding_model)
        self.retriever = None

    async def build_index(self, documents: List[Tuple[str, str]], chunk_size: int = 512):
        """Build the complete hybrid index."""
        print("Step 1: Creating basic text nodes...")
        for text, doc_id in documents:
            self.index_builder.add_document(text, doc_id, chunk_size=chunk_size)

        print(f"Created {len(self.index_builder.nodes)} nodes")

        print("Step 2: Extracting propositions...")
        await self.index_builder.extract_propositions(self.extractor)

        stats = self.index_builder.get_statistics()
        print(f"Extracted {stats['total_propositions']} propositions")
        print(".2f")

        print("Step 3: Building vector index...")
        self.index_builder.build_vector_index()

        print("Step 4: Initializing recursive retriever...")
        self.retriever = RecursivePropositionRetriever(self.index_builder)

        print("Index building complete!")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve using the proposition-based system."""
        if not self.retriever:
            raise RuntimeError("Index not built. Call build_index first.")

        return self.retriever.retrieve(query, top_k)

    def get_index_stats(self) -> Dict[str, Any]:
        """Get comprehensive index statistics."""
        return self.index_builder.get_statistics()


class PerformanceEvaluator:
    """Evaluates proposition retrieval performance."""

    def __init__(self):
        self.metrics = {}

    def evaluate_retrieval(self, queries: List[str], ground_truth: List[List[str]],
                          pipeline: PropositionRetrievalPipeline) -> Dict[str, float]:
        """Evaluate retrieval performance."""
        precision_scores = []
        recall_scores = []
        proposition_hit_rates = []

        for query, gt_docs in zip(queries, ground_truth):
            results = pipeline.retrieve(query, top_k=10)

            # Extract retrieved document IDs
            retrieved_ids = set()
            proposition_hits = 0

            for result in results:
                if result["type"] == "proposition":
                    # Map proposition back to parent node
                    parent_id = result["metadata"].get("parent_id")
                    if parent_id:
                        retrieved_ids.add(parent_id)
                    proposition_hits += 1
                else:
                    retrieved_ids.add(result["id"])

            # Calculate metrics
            gt_set = set(gt_docs)
            retrieved_set = retrieved_ids

            if retrieved_set:
                precision = len(gt_set & retrieved_set) / len(retrieved_set)
            else:
                precision = 0.0

            recall = len(gt_set & retrieved_set) / len(gt_set) if gt_set else 1.0

            precision_scores.append(precision)
            recall_scores.append(recall)
            proposition_hit_rates.append(proposition_hits / len(results) if results else 0)

        return {
            "avg_precision": np.mean(precision_scores),
            "avg_recall": np.mean(recall_scores),
            "avg_f1": 2 * np.mean(precision_scores) * np.mean(recall_scores) / (np.mean(precision_scores) + np.mean(recall_scores) + 1e-8),
            "proposition_hit_rate": np.mean(proposition_hit_rates)
        }


if __name__ == "__main__":
    async def main():
        # Example documents
        documents = [
            ("New Jersey has several excellent hot springs. Shunjing Hot Spring is well-known for its resort facilities. Jiuhua Mountain Resort offers scenic spa services. The Capital Hot Spring provides popular hotel amenities.",
             "doc1"),
            ("To reach the spa resort, take the main highway and follow the signs. The resort is located in a beautiful mountain area with easy access from downtown.",
             "doc2"),
            ("This hot spring is famous because it has unique mineral content that provides therapeutic benefits. Unlike regular spas, it offers authentic hot spring experience with natural geothermal water.",
             "doc3")
        ]

        # Create pipeline
        pipeline = PropositionRetrievalPipeline()

        # Build index
        await pipeline.build_index(documents)

        # Test retrieval
        queries = [
            "What are the best hot springs in New Jersey?",
            "How do I get to the spa resort?",
            "Why is this hot spring famous?"
        ]

        print("\n=== Proposition-Based Retrieval Results ===")
        for query in queries:
            print(f"\nQuery: {query}")
            results = pipeline.retrieve(query, top_k=3)

            for i, result in enumerate(results, 1):
                print(f"{i}. [{result['type']}] {result['text'][:100]}... (score: {result['score']:.3f})")

        # Performance evaluation
        ground_truth = [
            ["doc1"],  # Best hot springs -> doc1
            ["doc2"],  # How to get to spa -> doc2
            ["doc3"]   # Why famous -> doc3
        ]

        evaluator = PerformanceEvaluator()
        metrics = evaluator.evaluate_retrieval(queries, ground_truth, pipeline)

        print("\n=== Performance Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")

    # Run example
    asyncio.run(main())

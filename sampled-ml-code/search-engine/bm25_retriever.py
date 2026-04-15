import math
from collections import Counter
import heapq
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from document_processor import TextProcessor


TokenizedQuery = Union[str, List[str]]


class BM25:
    """BM25 sparse retrieval algorithm implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[List[str]] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.avg_doc_len = 0.0
        self.doc_len: List[int] = []
        self.num_docs = 0
        self.vocab: set = set()

    def index(self, documents: List[List[str]]) -> None:
        self.docs = []
        self.doc_freqs = []
        self.idf = {}
        self.avg_doc_len = 0.0
        self.doc_len = []
        self.num_docs = 0
        self.vocab = set()

        self.docs = documents
        self.num_docs = len(documents)

        doc_counts = Counter()
        for doc in documents:
            term_counts = Counter(doc)
            self.doc_freqs.append(term_counts)
            self.doc_len.append(len(doc))
            self.vocab.update(term_counts)
            doc_counts.update(term_counts.keys())

        self.avg_doc_len = sum(self.doc_len) / self.num_docs if self.num_docs > 0 else 0.0

        for term, doc_count in doc_counts.items():
            self.idf[term] = math.log((self.num_docs - doc_count + 0.5) / (doc_count + 0.5) + 1)

    def search(self, query: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        if not self.docs or not query:
            return []

        query_freq = Counter(query)
        scores = []

        for doc_id, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            for term in query_freq:
                if term not in self.idf:
                    continue

                term_freq = doc_freq.get(term, 0)
                if term_freq == 0:
                    continue

                idf = self.idf[term]
                doc_len = self.doc_len[doc_id]
                numerator = idf * term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1e-8)))
                score += numerator / denominator

            if score > 0:
                scores.append((doc_id, score))

        return heapq.nlargest(top_k, scores, key=lambda item: item[1])

    def batch_search(self, queries: List[List[str]], top_k: int = 5) -> List[List[Tuple[int, float]]]:
        return [self.search(query, top_k) for query in queries]


class BM25Retriever:
    """High-level BM25 retriever for preprocessed documents and segments."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, text_processor: Optional[TextProcessor] = None):
        self.bm25 = BM25(k1=k1, b=b)
        self.text_processor = text_processor or TextProcessor()
        self.documents: List[Dict[str, object]] = []
        self.document_tokens: List[List[str]] = []

    def _normalize_document(
        self,
        document: Any,
        text_fields: Optional[Sequence[str]],
    ) -> Tuple[Dict[str, object], List[str]]:
        if isinstance(document, dict):
            doc_copy = document.copy()
            normalized_text = self.text_processor.normalize_document(document, text_fields=text_fields)
            doc_copy.setdefault("content", normalized_text)
            return doc_copy, self.text_processor.tokenize_normalized(normalized_text.lower())

        normalized_text = self.text_processor.normalize_document(document)
        return {"content": normalized_text}, self.text_processor.tokenize_normalized(normalized_text.lower())

    def index_documents(self, documents: List[Any], text_fields: Optional[Sequence[str]] = None) -> None:
        self.documents = []
        self.document_tokens = []

        normalized_documents = []
        for document in documents:
            doc_copy, tokens = self._normalize_document(document, text_fields)
            normalized_documents.append(doc_copy)
            self.document_tokens.append(tokens)

        self.documents = normalized_documents
        self.bm25.index(self.document_tokens)

    def _normalize_query(self, query: TokenizedQuery) -> List[str]:
        if isinstance(query, str):
            return self.text_processor.tokenize(query)
        return [token.lower() for token in query if token]

    def retrieve(self, query: TokenizedQuery, top_k: int = 5) -> List[Dict[str, object]]:
        query_tokens = self._normalize_query(query)
        results = self.bm25.search(query_tokens, top_k=top_k)

        retrieved_docs = []
        for rank, (doc_id, score) in enumerate(results, start=1):
            document = self.documents[doc_id].copy()
            matched_terms = sorted(set(query_tokens) & set(self.document_tokens[doc_id]))
            document["bm25_score"] = score
            document["matched_terms"] = matched_terms
            document["rank"] = rank
            retrieved_docs.append(document)

        return retrieved_docs

    def batch_retrieve(self, queries: List[TokenizedQuery], top_k: int = 5) -> List[List[Dict[str, object]]]:
        return [self.retrieve(query, top_k) for query in queries]


if __name__ == "__main__":
    documents = [
        {
            "id": 1,
            "name": "Shunjing Hot Spring",
            "category": "Hot Spring",
            "description": "Well-known resort with scenic outdoor pools.",
        },
        {
            "id": 2,
            "name": "Jiuhua Mountain Resort",
            "category": "Hot Spring",
            "description": "Spa resort with mountain views and family amenities.",
        },
        {
            "id": 3,
            "rec_texts": [
                "City Spa Center offers relaxing services.",
                "Contact us at spa@example.com for package details.",
            ],
        },
    ]

    retriever = BM25Retriever()
    retriever.index_documents(documents, text_fields=["name", "category", "description"])

    query = "hot spring resort"
    results = retriever.retrieve(query, top_k=2)
    print(f"Query: {query}")
    for doc in results:
        label = doc.get("name", doc.get("content", "document"))
        print(
            f"  - {label} "
            f"(BM25 score: {doc['bm25_score']:.4f}, matched_terms={doc['matched_terms']})"
        )

    batch_queries = ["hot spring", ["spa", "services"]]
    batch_results = retriever.batch_retrieve(batch_queries, top_k=2)
    for query_item, query_results in zip(batch_queries, batch_results):
        print(f"Query: {query_item}")
        for doc in query_results:
            label = doc.get("name", doc.get("content", "document"))
            print(f"  - {label} (BM25 score: {doc['bm25_score']:.4f})")

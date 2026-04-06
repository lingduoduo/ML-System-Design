import math
from typing import List, Dict, Tuple
from collections import Counter


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
        self.docs = documents
        self.num_docs = len(documents)
        
        # Calculate document frequencies
        for doc in documents:
            self.doc_freqs.append(Counter(doc))
            self.doc_len.append(len(doc))
            self.vocab.update(doc)
        
        # Calculate average document length
        self.avg_doc_len = sum(self.doc_len) / self.num_docs if self.num_docs > 0 else 0
        
        # Calculate IDF for each term
        for term in self.vocab:
            doc_count = sum(1 for doc_freq in self.doc_freqs if term in doc_freq)
            self.idf[term] = math.log((self.num_docs - doc_count + 0.5) / (doc_count + 0.5) + 1)

    def search(self, query: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        if not self.docs:
            return []
        
        query_freq = Counter(query)
        scores = []
        
        for doc_id, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            for term in query_freq:
                if term not in self.idf:
                    continue
                
                term_freq = doc_freq.get(term, 0)
                idf = self.idf[term]
                doc_len = self.doc_len[doc_id]
                
                numerator = idf * term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += numerator / denominator
            
            if score > 0:
                scores.append((doc_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def batch_search(self, queries: List[List[str]], top_k: int = 5) -> List[List[Tuple[int, float]]]:
        return [self.search(query, top_k) for query in queries]


class BM25Retriever:
    """High-level BM25 retriever for POI/document ranking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.bm25 = BM25(k1=k1, b=b)
        self.documents: List[Dict[str, object]] = []

    def index_documents(self, documents: List[Dict[str, object]], text_fields: List[str] = None) -> None:
        if text_fields is None:
            text_fields = ["name", "description", "category"]
        
        self.documents = documents
        indexed_docs = []
        
        for doc in documents:
            tokens = []
            for field in text_fields:
                text = str(doc.get(field, "")).lower().split()
                tokens.extend(text)
            indexed_docs.append(tokens)
        
        self.bm25.index(indexed_docs)

    def retrieve(self, query: List[str], top_k: int = 5) -> List[Dict[str, object]]:
        query_lower = [token.lower() for token in query]
        results = self.bm25.search(query_lower, top_k=top_k)
        
        retrieved_docs = []
        for doc_id, score in results:
            doc = self.documents[doc_id].copy()
            doc["bm25_score"] = score
            retrieved_docs.append(doc)
        
        return retrieved_docs

    def batch_retrieve(self, queries: List[List[str]], top_k: int = 5) -> List[List[Dict[str, object]]]:
        return [self.retrieve(query, top_k) for query in queries]


if __name__ == "__main__":
    # Example with POI documents
    pois = [
        {"id": 1, "name": "Shunjing Hot Spring", "category": "Hot Spring", "description": "well-known resort"},
        {"id": 2, "name": "Jiuhua Mountain Resort", "category": "Hot Spring", "description": "scenic spa resort"},
        {"id": 3, "name": "City Spa Center", "category": "Spa", "description": "relaxing services"},
        {"id": 4, "name": "Capital Hot Spring", "category": "Hot Spring", "description": "popular hotel"},
    ]

    retriever = BM25Retriever()
    retriever.index_documents(pois, text_fields=["name", "category", "description"])

    # Retrieve top-2 for a query
    query = ["hot", "spring"]
    results = retriever.retrieve(query, top_k=2)
    print(f"Query: {query}")
    for doc in results:
        print(f"  - {doc['name']} (BM25 score: {doc['bm25_score']:.4f})")

    # Batch retrieval
    queries = [["hot", "spring"], ["spa"]]
    batch_results = retriever.batch_retrieve(queries, top_k=2)
    for query, results in zip(queries, batch_results):
        print(f"Query: {query}")
        for doc in results:
            print(f"  - {doc['name']} (BM25 score: {doc['bm25_score']:.4f})")

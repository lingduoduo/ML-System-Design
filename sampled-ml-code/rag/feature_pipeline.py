from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import logging
from math import log
from typing import Any, Dict, Iterable, List, Literal, Tuple

import numpy as np


RetrievalMode = Literal["dense", "bm25", "hnsw"]
RetrievalResult = List[Tuple[float, Dict]]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


DOCUMENT_PATH = "./data/"
FILE_PATTERN = "*.md"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_LLM_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 512


def chunk_text(text: str, chunk_size: int = 50) -> List[str]:
    words = text.split()
    return [" ".join(words[index:index + chunk_size]) for index in range(0, len(words), chunk_size)]


def tokenize(text: str) -> List[str]:
    normalized = text.lower()
    return [token.strip(".,!?;:\"'()[]{}") for token in normalized.split() if token.strip(".,!?;:\"'()[]{}")]


def _import_langchain_dependencies() -> Dict[str, Any]:
    try:
        from langchain_community.document_loaders import DirectoryLoader
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise ImportError(
            "LangChain dependencies are not installed. "
            "Install `langchain-community`, `langchain-text-splitters`, "
            "`faiss-cpu`, `sentence-transformers`, and `transformers` to use "
            "the FAISS/HuggingFace feature pipeline."
        ) from exc

    return {
        "DirectoryLoader": DirectoryLoader,
        "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
        "FAISS": FAISS,
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
    }


def _format_docs(docs: List[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_embeddings(model_name: str = EMBEDDING_MODEL) -> Any:
    deps = _import_langchain_dependencies()
    logging.info("Loading HF embeddings: %s", model_name)
    return deps["HuggingFaceEmbeddings"](
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_local_hf_llm() -> Any:
    logging.info(f"Loading local HF LLM: {HF_LLM_MODEL_ID}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from langchain_community.llms import HuggingFacePipeline
    except ImportError as exc:
        raise ImportError(
            "Transformers dependencies are not installed. "
            "Install `torch`, `transformers`, and `langchain-community` "
            "to use the local Hugging Face LLM pipeline."
        ) from exc

    use_cuda = torch.cuda.is_available()
    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    tokenizer = AutoTokenizer.from_pretrained(HF_LLM_MODEL_ID, use_fast=True)

    if use_cuda:
        logging.info("Using CUDA GPU")
        model = AutoModelForCausalLM.from_pretrained(
            HF_LLM_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        dtype = torch.float16 if use_mps else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            HF_LLM_MODEL_ID,
            torch_dtype=dtype,
        )
        if use_mps:
            logging.info("Using Apple MPS")
            model = model.to("mps")
        else:
            logging.info("Using CPU (slow)")

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        repetition_penalty=1.05,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)


@dataclass
class LangChainFeatureStore:
    documents: List[Any]
    chunks: List[Any]
    embeddings: Any
    vector_store: Any
    retriever: Any


def build_langchain_feature_store(
    document_path: str = DOCUMENT_PATH,
    file_pattern: str = FILE_PATTERN,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    top_k: int = TOP_K,
    embedding_model: str = EMBEDDING_MODEL,
) -> LangChainFeatureStore:
    deps = _import_langchain_dependencies()
    loader = deps["DirectoryLoader"](document_path, glob=file_pattern)
    documents = loader.load()
    logging.info("Loaded %s source documents from %s", len(documents), document_path)

    splitter = deps["RecursiveCharacterTextSplitter"](
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logging.info("Split source documents into %s chunks", len(chunks))

    embeddings = build_embeddings(embedding_model)
    vector_store = deps["FAISS"].from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    return LangChainFeatureStore(
        documents=documents,
        chunks=chunks,
        embeddings=embeddings,
        vector_store=vector_store,
        retriever=retriever,
    )


@dataclass
class QueryExpander:
    synonym_map: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "rag": ["retrieval augmented generation", "retrieval"],
            "retrieve": ["search", "lookup"],
            "retrieval": ["search", "document retrieval"],
            "llm": ["language model", "large language model"],
            "pipeline": ["workflow", "system"],
            "docs": ["documents", "knowledge base"],
        }
    )
    max_expansions: int = 3

    def expand(self, query: str) -> List[str]:
        expansions = [query]
        tokens = tokenize(query)

        for token in tokens:
            for synonym in self.synonym_map.get(token, []):
                expansions.append(self._replace_token(query, token, synonym))
                if len(expansions) >= self.max_expansions + 1:
                    return self._dedupe(expansions)

        return self._dedupe(expansions)

    def _replace_token(self, query: str, token: str, synonym: str) -> str:
        query_tokens = query.split()
        replaced = [synonym if raw_token.lower().strip(".,!?;:\"'()[]{}") == token else raw_token for raw_token in query_tokens]
        return " ".join(replaced)

    def _dedupe(self, queries: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for candidate in queries:
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped


@dataclass
class SimpleEmbedder:
    dimension: int = 8
    _cache: Dict[str, np.ndarray] = field(default_factory=dict)

    def embed(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        embedding = rng.random(self.dimension)
        self._cache[text] = embedding
        return embedding


@dataclass
class VectorDB:
    rows: List[Dict] = field(default_factory=list)
    _matrix: np.ndarray | None = None
    _tokenized_docs: List[List[str]] = field(default_factory=list)
    _doc_term_freqs: List[Counter[str]] = field(default_factory=list)
    _doc_freqs: Dict[str, int] = field(default_factory=dict)
    _avg_doc_length: float = 0.0
    _hnsw_graph: Dict[int, List[int]] = field(default_factory=dict)

    def add(self, vector: np.ndarray, metadata: Dict) -> None:
        index = len(self.rows)
        tokens = tokenize(metadata["text"])
        term_freq = Counter(tokens)

        self.rows.append({"vector": vector, "metadata": metadata})
        self._tokenized_docs.append(tokens)
        self._doc_term_freqs.append(term_freq)
        self._matrix = None

        for token in term_freq:
            self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1

        total_length = self._avg_doc_length * index + len(tokens)
        self._avg_doc_length = total_length / len(self.rows)
        self._link_hnsw_node(index)

    def _link_hnsw_node(self, index: int, max_neighbors: int = 3) -> None:
        self._hnsw_graph.setdefault(index, [])
        if index == 0:
            return

        vector = self.rows[index]["vector"]
        similarities = [
            (float(vector @ row["vector"]), neighbor_index)
            for neighbor_index, row in enumerate(self.rows[:-1])
        ]
        nearest = [neighbor_index for _, neighbor_index in sorted(similarities, reverse=True)[:max_neighbors]]
        self._hnsw_graph[index] = nearest

        for neighbor_index in nearest:
            neighbor_edges = self._hnsw_graph.setdefault(neighbor_index, [])
            if index not in neighbor_edges:
                neighbor_edges.append(index)

    def ensure_matrix(self) -> np.ndarray:
        if self._matrix is None:
            if not self.rows:
                self._matrix = np.empty((0, 0))
            else:
                self._matrix = np.vstack([row["vector"] for row in self.rows])
        return self._matrix

    def top_indices_from_scores(self, scores: np.ndarray, top_k: int) -> np.ndarray:
        limit = min(top_k, len(scores))
        if limit <= 0:
            return np.array([], dtype=int)
        if limit == len(scores):
            return np.argsort(scores)[::-1]

        partition = np.argpartition(scores, -limit)[-limit:]
        return partition[np.argsort(scores[partition])[::-1]]


@dataclass
class DenseRetriever:
    embedder: SimpleEmbedder
    vector_db: VectorDB

    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        if not self.vector_db.rows:
            return []

        query_vector = self.embedder.embed(query)
        scores = self.vector_db.ensure_matrix() @ query_vector
        top_indices = self.vector_db.top_indices_from_scores(scores, top_k)
        return [(float(scores[index]), self.vector_db.rows[index]["metadata"]) for index in top_indices]


@dataclass
class BM25Retriever:
    vector_db: VectorDB
    k1: float = 1.5
    b: float = 0.75

    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        if not self.vector_db.rows:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        total_docs = len(self.vector_db.rows)
        scores = np.zeros(total_docs, dtype=float)

        for index, term_freq in enumerate(self.vector_db._doc_term_freqs):
            doc_length = max(1, len(self.vector_db._tokenized_docs[index]))
            norm = self.k1 * (1 - self.b + self.b * doc_length / max(self.vector_db._avg_doc_length, 1.0))
            score = 0.0
            for token in query_tokens:
                freq = term_freq.get(token, 0)
                if freq == 0:
                    continue

                doc_freq = self.vector_db._doc_freqs.get(token, 0)
                idf = log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                score += idf * ((freq * (self.k1 + 1)) / (freq + norm))
            scores[index] = score

        top_indices = self.vector_db.top_indices_from_scores(scores, top_k)
        return [
            (float(scores[index]), self.vector_db.rows[index]["metadata"])
            for index in top_indices
            if scores[index] > 0
        ]


@dataclass
class HNSWRetriever:
    embedder: SimpleEmbedder
    vector_db: VectorDB
    ef_search: int = 6

    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        if not self.vector_db.rows:
            return []

        query_vector = self.embedder.embed(query)
        entrypoint = 0
        visited = {entrypoint}
        frontier = [entrypoint]
        candidate_scores: Dict[int, float] = {}

        while frontier and len(visited) < max(self.ef_search, top_k):
            current = frontier.pop()
            candidate_scores[current] = float(self.vector_db.rows[current]["vector"] @ query_vector)

            neighbors = self.vector_db._hnsw_graph.get(current, [])
            ranked_neighbors = sorted(
                neighbors,
                key=lambda neighbor_index: float(self.vector_db.rows[neighbor_index]["vector"] @ query_vector),
                reverse=True,
            )
            for neighbor_index in ranked_neighbors:
                if neighbor_index in visited:
                    continue
                visited.add(neighbor_index)
                frontier.append(neighbor_index)
                if len(visited) >= self.ef_search:
                    break

        if len(candidate_scores) < top_k:
            return DenseRetriever(embedder=self.embedder, vector_db=self.vector_db).search(query, top_k=top_k)

        ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [(score, self.vector_db.rows[index]["metadata"]) for index, score in ranked]


@dataclass
class RetrieverIndex:
    dense: DenseRetriever
    bm25: BM25Retriever
    hnsw: HNSWRetriever
    expander: QueryExpander | None = None

    @classmethod
    def from_components(cls, embedder: SimpleEmbedder, vector_db: VectorDB) -> RetrieverIndex:
        return cls(
            dense=DenseRetriever(embedder=embedder, vector_db=vector_db),
            bm25=BM25Retriever(vector_db=vector_db),
            hnsw=HNSWRetriever(embedder=embedder, vector_db=vector_db),
            expander=QueryExpander(),
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
        mode: RetrievalMode = "dense",
        expand_query: bool = False,
    ) -> RetrievalResult:
        candidates = self.expander.expand(query) if expand_query and self.expander is not None else [query]
        aggregated: Dict[Tuple[str, int], Tuple[float, Dict]] = {}

        for candidate in candidates:
            if mode == "bm25":
                results = self.bm25.search(candidate, top_k=top_k)
            elif mode == "hnsw":
                results = self.hnsw.search(candidate, top_k=top_k)
            else:
                results = self.dense.search(candidate, top_k=top_k)

            for score, metadata in results:
                key = (metadata["doc_id"], metadata["chunk_id"])
                current = aggregated.get(key)
                if current is None or score > current[0]:
                    aggregated[key] = (score, metadata)

        ranked = sorted(aggregated.values(), key=lambda item: item[0], reverse=True)
        return ranked[:top_k]


def build_feature_store(
    docs: Iterable[Dict],
    embedder: SimpleEmbedder | None = None,
    chunk_size: int = 10,
) -> tuple[SimpleEmbedder, VectorDB, List[Dict]]:
    embedder = embedder or SimpleEmbedder()
    vector_db = VectorDB()
    instruct_dataset: List[Dict] = []

    for doc in docs:
        for chunk_id, chunk in enumerate(chunk_text(doc["content"], chunk_size=chunk_size)):
            metadata = {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "chunk_id": chunk_id,
                "text": chunk,
            }
            vector_db.add(embedder.embed(chunk), metadata)
            instruct_dataset.append(
                {
                    "instruction": f"Summarize this chunk from {doc['title']}",
                    "input": chunk,
                    "output": f"Summary placeholder for: {chunk}",
                }
            )

    return embedder, vector_db, instruct_dataset
